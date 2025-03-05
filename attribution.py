from collections import namedtuple
import nnsight
import nnsight.tracing
import torch as t
from tqdm import tqdm
from numpy import ndindex
from loading_utils import Submodule
from activation_utils import SparseAct
from nnsight.envoy import Envoy
from dictionary_learning.dictionary import Dictionary, JumpReluAutoEncoder
from typing import Callable, Dict, List, Optional
import types

EffectOut = namedtuple(
    'EffectOut', ['effects', 'deltas', 'grads', 'total_effect'])


def create_batches(input_data, batch_size):
  return [
      input_data[i:i + batch_size]
      for i in range(0, len(input_data), batch_size)
  ]


def get_tokens_and_length(
    model: nnsight.LanguageModel,
    input_data: list[str],
) -> tuple[list[str], list[int]]:
  tokens = model.tokenizer(input_data, padding=False)
  lengths = [len(token) - 1 for token in tokens]
  tokens = model.tokenizer.pad(tokens)
  return tokens, lengths


def get_activations(
    model: nnsight.LanguageModel,
    input_data: list[str],
    submods: list[Submodule],
    SAE: dict[Submodule, Dictionary],
    batch_size: int = 32,
    aggregation: str = 'mean',
    calc_metric: bool = False,
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
  acts = dict()
  metrics = list()

  # Batch input data for memory requirements
  input_data = create_batches(input_data, batch_size)

  for batch in input_data:
    tokens = model.tokenizer(batch, padding=True, return_tensors='pt')
    with t.no_grad(), model.trace(tokens):
      for submod in submods:
        x = submod.get_activation()[:, -1, ...].unsqueeze(1)

        # Pass activation through SAE to get feature activations and error.
        x_hat, f = SAE[submod](x, output_features=True)
        residual = x - x_hat
        if aggregation == 'none':
          batch_act = SparseAct(act=f, res=residual)
          acts[submod] = acts.get(submod, []) + [batch_act]
        else:
          batch_act = SparseAct(act=f, res=residual).sum(dim=0)
          if submod in acts:
            acts[submod] += batch_act
          else:
            acts[submod] = batch_act
        acts[submod] = acts[submod].save()
      if calc_metric:
        metrics.append(metric_fn(model, **metric_kwargs).sum(dim=0).save())

  metric = t.mean(t.stack([m.value for m in metrics])) if calc_metric else None
  acts = {submod: act.value for submod, act in acts.items()}
  if aggregation == 'mean':
    for submod in submods:
      acts[submod] /= len(input_data)
  elif aggregation == 'none':
    for submod in submods:
      acts[submod] = t.cat(acts[submod], dim=0)

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
) -> t.Tensor:
  """
  Run the model with patched activations once.

  Args:
      model: The model to run the trace on
  Returns:
      Metric value for the model with the patched activations
  """
  input_data = create_batches(input_data, batch_size)
  metric = 0

  for batch in input_data:
    tokens = model.tokenizer(batch, padding=True, return_tensors='pt')
    with t.no_grad(), model.trace(tokens):
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
        clean[:, -1, ...] = x_patch
      metric += metric_fn(model, **metric_kwargs).sum(dim=0)
      metric.save()
  return metric / len(input_data)


def pe_exact(
    model: nnsight.LanguageModel,
    clean_input: list[str],
    clean_acts: dict[Submodule, SparseAct],
    patch_acts: dict[Submodule, SparseAct],
    submods: list[Submodule],
    SAE: dict[Submodule, Dictionary],
    metric_fn: Callable,
    metric_kwargs: dict = dict(),
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
      )

    effects[submod] = effect
    deltas[submod] = patch_act - clean_act

  return EffectOut(effects, deltas, None, None)


def _pe_attrib(
    clean,
    patch,
    model,
    submodules: List[Submodule],
    dictionaries: Dict[Submodule, Dictionary],
    metric_fn,
    metric_kwargs=dict(),
):
  # Get clean activations with gradients
  hidden_states_clean = {}
  grads = {}
  with model.trace(clean):
    for submodule in submodules:
      dictionary = dictionaries[submodule]
      x = submodule.get_activation()[:, -1, :].unsqueeze(1)
      # x_hat implicitly depends on f
      x_hat, f = dictionary(x, output_features=True)
      residual = x - x_hat
      hidden_states_clean[submodule] = SparseAct(
          act=f, res=residual).save()
      grads[submodule] = hidden_states_clean[submodule].grad.save()
      residual.grad = t.zeros_like(residual)
      x_recon = x_hat + residual
      submodule.set_activation(x_recon)
      x.grad = x_recon.grad
    metric_clean = metric_fn(model, **metric_kwargs).save()
    metric_clean.sum().backward()
  hidden_states_clean = {k: v.value for k, v in hidden_states_clean.items()}
  grads = {k: v.value for k, v in grads.items()}

  # Handle patch inputs
  if patch is None:
    hidden_states_patch = _handle_patch_none_case(hidden_states_clean)
    total_effect = None
  else:
    hidden_states_patch = {}
    with t.no_grad(), model.trace(patch):
      for submodule in submodules:
        dictionary = dictionaries[submodule]
        x = submodule.get_activation()[:, -1, :].unsqueeze(1)
        x_hat, f = dictionary(x, output_features=True)
        residual = x - x_hat
        hidden_states_patch[submodule] = SparseAct(
            act=f, res=residual).save()
      metric_patch = metric_fn(model, **metric_kwargs).save()
    total_effect = (metric_patch.value - metric_clean.value).detach()
    hidden_states_patch = {k: v.value for k,
                           v in hidden_states_patch.items()}

  # Calculate effects
  effects = {}
  deltas = {}
  for submodule in submodules:
    patch_state, clean_state, grad = hidden_states_patch[
        submodule], hidden_states_clean[submodule], grads[submodule]
    delta = patch_state - clean_state.detach() if patch_state is not None else - \
        clean_state.detach()
    effect = delta @ grad
    effects[submodule] = effect
    deltas[submodule] = delta

  return EffectOut(effects, deltas, grads, total_effect)


def _pe_ig(
    clean,
    patch,
    model,
    submodules: List[Submodule],
    dictionaries: Dict[Submodule, Dictionary],
    metric_fn,
    steps=10,
    metric_kwargs=dict(),
):
  # Get clean activations
  hidden_states_clean = {}
  with t.no_grad(), model.trace(clean):
    for submodule in submodules:
      dictionary = dictionaries[submodule]
      x = submodule.get_activation()[:, -1, :].unsqueeze(1)
      f = dictionary.encode(x)
      x_hat = dictionary.decode(f)
      residual = x - x_hat
      hidden_states_clean[submodule] = SparseAct(
          act=f.save(), res=residual.save())
    metric_clean = metric_fn(model, **metric_kwargs).save()
  hidden_states_clean = {k: v.value for k, v in hidden_states_clean.items()}

  # Handle patch inputs
  if patch is None:
    hidden_states_patch = _handle_patch_none_case(hidden_states_clean)
    total_effect = None
  else:
    hidden_states_patch = {}
    with t.no_grad(), model.trace(patch):
      for submodule in submodules:
        dictionary = dictionaries[submodule]
        x = submodule.get_activation()[:, -1, :].unsqueeze(1)
        f = dictionary.encode(x)
        x_hat = dictionary.decode(f)
        residual = x - x_hat
        hidden_states_patch[submodule] = SparseAct(
            act=f.save(), res=residual.save())
      metric_patch = metric_fn(model, **metric_kwargs).save()
    total_effect = (metric_patch.value - metric_clean.value).detach()
    hidden_states_patch = {k: v.value for k,
                           v in hidden_states_patch.items()}

  # Calculate integrated gradients
  effects = {}
  deltas = {}
  grads = {}
  for submodule in submodules:
    dictionary = dictionaries[submodule]
    clean_state = hidden_states_clean[submodule]
    patch_state = hidden_states_patch[submodule]
    with model.trace() as tracer:
      metrics = []
      fs = []
      for step in range(steps):
        alpha = step / steps
        f = (1 - alpha) * clean_state + alpha * patch_state
        f.act.requires_grad_().retain_grad()
        f.res.requires_grad_().retain_grad()
        fs.append(f)
        with tracer.invoke(clean):
          submodule.set_activation(dictionary.decode(f.act) + f.res)
          metrics.append(metric_fn(model, **metric_kwargs))
      metric = sum([m for m in metrics])
      metric.sum().backward()

    mean_grad = sum([f.act.grad for f in fs]) / steps
    mean_residual_grad = sum([f.res.grad for f in fs]) / steps
    grad = SparseAct(act=mean_grad, res=mean_residual_grad)
    delta = (
        patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
    effect = grad @ delta

    effects[submodule] = effect
    deltas[submodule] = delta
    grads[submodule] = grad

  return EffectOut(effects, deltas, grads, total_effect)


def _pe_exact(
    clean,
    patch,
    model,
    submodules: List[Submodule],
    dictionaries: Dict[Submodule, Dictionary],
    metric_fn,
    metric_kwargs=dict(),
):
  # Get clean activations
  hidden_states_clean = {}
  with t.no_grad(), model.trace(clean):
    for submodule in submodules:
      dictionary = dictionaries[submodule]
      x = submodule.get_activation()[:, -1, :].unsqueeze(1)
      f = dictionary.encode(x)
      x_hat = dictionary.decode(f)
      residual = x - x_hat
      hidden_states_clean[submodule] = SparseAct(
          act=f, res=residual).save()
    metric_clean = metric_fn(model, **metric_kwargs).save()
  hidden_states_clean = {k: v.value for k, v in hidden_states_clean.items()}

  # Handle patch inputs
  if patch is None:
    hidden_states_patch = _handle_patch_none_case(hidden_states_clean)
    total_effect = None
  else:
    hidden_states_patch = {}
    with t.no_grad(), model.trace(patch):
      for submodule in submodules:
        dictionary = dictionaries[submodule]
        x = submodule.get_activation()[:, -1, :].unsqueeze(1)
        f = dictionary.encode(x)
        x_hat = dictionary.decode(f)
        residual = x - x_hat
        hidden_states_patch[submodule] = SparseAct(
            act=f, res=residual).save()
      metric_patch = metric_fn(model, **metric_kwargs).save()
    total_effect = metric_patch.value - metric_clean.value
    hidden_states_patch = {k: v.value for k,
                           v in hidden_states_patch.items()}

  # Calculate exact effects by evaluating each feature independently
  effects = {}
  deltas = {}
  for submodule in submodules:
    dictionary = dictionaries[submodule]
    clean_state = hidden_states_clean[submodule]
    patch_state = hidden_states_patch[submodule]
    effect = SparseAct(act=t.zeros_like(clean_state.act),
                       resc=t.zeros(*clean_state.res.shape[:-1])).to(model.device)

    # Iterate over positions and features for which clean and patch differ
    idxs = t.nonzero(patch_state.act - clean_state.act)
    for idx in tqdm(idxs):
      with t.no_grad(), model.trace(clean):
        f = clean_state.act.clone()
        f[tuple(idx)] = patch_state.act[tuple(idx)]
        x_hat = dictionary.decode(f)
        submodule.set_activation(x_hat + clean_state.res)
        metric = metric_fn(model, **metric_kwargs).save()
      effect.act[tuple(idx)] = (metric.value - metric_clean.value).sum()

    for idx in list(ndindex(effect.resc.shape)):
      with t.no_grad(), model.trace(clean):
        res = clean_state.res.clone()
        res[tuple(idx)] = patch_state.res[tuple(idx)]
        x_hat = dictionary.decode(clean_state.act)
        submodule.set_activation(x_hat + res)
        metric = metric_fn(model, **metric_kwargs).save()
      effect.resc[tuple(idx)] = (metric.value - metric_clean.value).sum()

    effects[submodule] = effect
    deltas[submodule] = patch_state - clean_state

  return EffectOut(effects, deltas, None, total_effect)


def patching_effect(
    clean,
    patch,
    model,
    submodules: List[Submodule],
    dictionaries: Dict[Submodule, Dictionary],
    metric_fn: Callable[[Envoy], t.Tensor],
    method='attrib',
    steps=10,
    metric_kwargs=dict()
):
  """
  Calculate patching effect using various methods.

  Args:
      clean: Clean input
      patch: Patch input (or None for ablation)
      model: Model to evaluate
      submodules: List of submodules to analyze
      dictionaries: Dictionary for each submodule
      metric_fn: Function to compute metric
      method: Method to use ('attrib', 'ig', or 'exact')
      steps: Number of steps for integrated gradients
      metric_kwargs: Additional kwargs for metric function

  Returns:
      EffectOut with effects, deltas, grads, and total_effect
  """
  if method == 'attrib':
    return _pe_attrib(clean, patch, model, submodules, dictionaries, metric_fn, metric_kwargs=metric_kwargs)
  elif method == 'ig':
    return _pe_ig(clean, patch, model, submodules, dictionaries, metric_fn, steps=steps, metric_kwargs=metric_kwargs)
  elif method == 'exact':
    return _pe_exact(clean, patch, model, submodules, dictionaries, metric_fn, metric_kwargs=metric_kwargs)
  else:
    raise ValueError(f"Unknown method {method}")


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
