from collections import namedtuple
import nnsight
import nnsight.tracing
import torch as t
from tqdm import tqdm
from numpy import ndindex
from loading_utils import Submodule
from activation_utils import SparseAct
from nnsight.intervention import Envoy
from dictionary_learning.dictionary import Dictionary, JumpReluAutoEncoder
from typing import Callable, Dict, List, Optional
import types

EffectOut = namedtuple(
    'EffectOut', ['effects', 'deltas', 'grads', 'total_effect'])


def tokenize(
    model: nnsight.LanguageModel,
    input_data: list[str],
    batch_size: int,
):
  # TODO: Add support for batching.
  tokens = model.tokenizer(input_data, padding=True, return_tensors='pt')
  return tokens


def get_activations(
    model: nnsight.LanguageModel,
    input_data: list[str],
    submods: list[Submodule],
    SAE: dict[Submodule, Dictionary],
    batch_size: int = 32,
    aggre_dim: tuple[int] | None = (0, 1),
    metric_fn: Callable = None,
    metric_kwargs: dict = dict(),
) -> tuple[dict[Submodule, SparseAct], Optional[t.Tensor]]:
  """
  Extract activations at the from model for given inputs.

  Args:
      model: The model to run the trace on
      input_data: Input data for the model
      submodules: List of submodules to extract activations from
      dictionaries: Dictionary to use for each submodule
      aggregation: Aggregation method for activations across inputs
      calc_metric: Whether to calculate the metric
      require_grad: Whether activations should require gradient
  Returns:
      Dictionary of hidden states for each submodule
  """
  # TODO: add support for batching

  acts = dict()
  input_data = tokenize(model, input_data, batch_size)

  with t.no_grad(), model.trace(input_data):
    for submod in submods:
      x = submod.get_activation()

      # Pass activation through SAE to get feature activations and error.
      x_hat, f = SAE[submod](x, output_features=True)
      residual = x - x_hat
      acts[submod] = SparseAct(act=f, res=residual).mean(dim=aggre_dim).save()

    if metric_fn is not None:
      metric = metric_fn(model, **metric_kwargs).mean(dim=0)
    else:
      metric = None

  # Get the values from the graph nodes.
  for k in acts:
    acts[k] = acts[k].value

  return acts, metric


def _run_patch_sparse_features(
    model: nnsight.LanguageModel,
    input_data: list[str],
    clean_acts: dict[Submodule, SparseAct],
    patch_acts: dict[Submodule, SparseAct],
    patches: list[tuple],
    SAE: dict[Submodule, Dictionary],
    metric_fn: Callable,
    metric_kwargs: dict = dict(),
    batch_size: int = 32,
    patch_pos: int | None = None,
) -> t.Tensor:
  """
  Run the model with patched activations once.

  Args:
      model: The model to run the trace on
  Returns:
      Metric value for the model with the patched activations
  """
  input_data = tokenize(model, input_data, batch_size)

  with t.no_grad(), model.trace(input_data):
    for submod, idx, is_res in patches:
      # clean is the activation in orignal space (N, L, D) where L is sequence length
      clean = submod.get_activation()

      # Sparse repr of clean hidden state at the last position.
      f = clean_acts[submod].act.clone()
      res = clean_acts[submod].res.clone()

      # Patch sparse repr.
      if is_res:
        res[idx] = patch_acts[submod].res[idx]
      else:
        f[idx] = patch_acts[submod].act[idx]
      x_patch = SAE[submod].decode(f) + res

      # Change hidden state at the last position.
      clean[:, patch_pos:, ...] = x_patch
    metric = metric_fn(model, **metric_kwargs).sum(dim=0).save()

  return metric


def pe_exact(
    model: nnsight.LanguageModel,
    clean_input: list[str],
    clean_acts: dict[Submodule, SparseAct],
    patch_acts: dict[Submodule, SparseAct],
    submods: list[Submodule],
    SAE: dict[Submodule, Dictionary],
    metric_fn: Callable[[Envoy], t.Tensor],
    metric_kwargs: dict = dict(),
    patch_pos: int | None = None,
) -> EffectOut:
  """
  Calculate the exact effect of patching.
  """
  # Calculate exact effects by evaluating each feature independently
  effects = {}
  deltas = {}
  for submod in tqdm(submods):
    clean_act = clean_acts[submod]
    patch_act = patch_acts[submod]
    effect = SparseAct(
        act=t.zeros_like(clean_act.act),
        resc=t.zeros(clean_act.res.shape[:-1])
    ).to(model.device)

    # Iterate over positions and features for which clean and patch differ
    idxs = t.nonzero(patch_act.act - clean_act.act)
    for idx in idxs:
      idx = tuple(idx)
      effect.act[idx] = _run_patch_sparse_features(
          model,
          clean_input,
          clean_acts,
          patch_acts,
          [(submod, idx, False)],
          SAE,
          metric_fn,
          metric_kwargs,
          patch_pos,
      )
    for idx in list(ndindex(effect.resc.shape)):
      effect.resc[idx] = _run_patch_sparse_features(
          model,
          clean_input,
          clean_acts,
          patch_acts,
          [(submod, idx, True)],
          SAE,
          metric_fn,
          metric_kwargs,
          patch_pos,
      )

    effects[submod] = effect
    deltas[submod] = patch_act - clean_act

  return EffectOut(effects, deltas, None, None)


def jvp(
    input,
    model,
    dictionaries,
    downstream_submod,
    downstream_features,
    upstream_submod,
    left_vec: SparseAct,
    right_vec: SparseAct,
    intermediate_stopgrads: List[Submodule] = [],
):
  """
  Calculate Jacobian-vector product.

  Args:
      input: Input to the model
      model: Model to evaluate
      dictionaries: Dictionary for each submodule
      downstream_submod: Downstream submodule
      downstream_features: Features in downstream submodule
      upstream_submod: Upstream submodule
      left_vec: Left vector for JVP
      right_vec: Right vector for JVP
      intermediate_stopgrads: List of submodules to stop gradient at

  Returns:
      Sparse tensor with JVP values
  """
  # Monkey patching to get around an nnsight bug
  for dictionary in dictionaries.values():
    if isinstance(dictionary, JumpReluAutoEncoder):
      def hacked_forward(self, x):
        W_enc, W_dec = self.W_enc.data, self.W_dec.data
        b_enc, b_dec = self.b_enc.data, self.b_dec.data

        # Hacking around an nnsight bug
        pre_jump = x @ W_enc + b_enc
        f = t.nn.ReLU()(pre_jump * (pre_jump > self.threshold))
        f = f * W_dec.norm(dim=1)

        f_normed = f / W_dec.norm(dim=1)
        x_hat = f_normed @ W_dec + b_dec

        return x_hat, f
    else:
      def hacked_forward(self, x):
        return self.forward(x, output_features=True)

    dictionary.hacked_forward = types.MethodType(
        hacked_forward, dictionary)

  downstream_dict, upstream_dict = dictionaries[downstream_submod], dictionaries[upstream_submod]
  b, s, n_feats = downstream_features.act.shape

  if t.all(downstream_features.to_tensor() == 0):
    return t.sparse_coo_tensor(
        t.zeros((2 * downstream_features.act.dim(), 0), dtype=t.long),
        t.zeros(0),
        size=(b, s, n_feats + 1, b, s, n_feats + 1)
    ).to(model.device)

  vjv_values = {}

  downstream_feature_idxs = downstream_features.to_tensor().nonzero()
  with model.trace(input):
    # Forward pass modifications
    x = upstream_submod.get_activation()
    x_hat, f = upstream_dict.hacked_forward(x)
    x_res = x - x_hat
    upstream_submod.set_activation(x_hat + x_res)
    upstream_act = SparseAct(act=f, res=x_res).save()

    y = downstream_submod.get_activation()
    y_hat, g = downstream_dict.hacked_forward(y)
    y_res = y - y_hat
    downstream_act = SparseAct(act=g, res=y_res)

    to_backprops = (left_vec @ downstream_act).to_tensor()

    for downstream_feat_idx in downstream_feature_idxs:
      # Stop grad
      for submodule in intermediate_stopgrads:
        submodule.stop_grad()
      x_res.grad = t.zeros_like(x_res.grad)

      vjv = (upstream_act.grad @ right_vec).to_tensor().double().cuda()
      to_backprops[tuple(downstream_feat_idx)].backward(
          retain_graph=True)
      vjv_values[downstream_feat_idx] = vjv.save()

  vjv_indices = t.stack(list(vjv_values.keys()), dim=0).T
  vjv_values = t.stack([v.value[:, -1, :].unsqueeze(1)
                       for v in vjv_values.values()], dim=0)

  return t.sparse_coo_tensor(vjv_indices, vjv_values, size=(b, s, n_feats + 1, b, s, n_feats + 1))
