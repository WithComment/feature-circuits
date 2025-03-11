import nnsight.tracing
import nnsight.tracing.contexts
import torch as t
import nnsight
from nnsight.intervention import Envoy
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import Callable

from loading_utils import Submodule


gen_config = {
    'max_new_tokens': 100,
    'repetition_penalty': 1.2,
}


def get_acts_met_embedding(
    model: nnsight.LanguageModel,
    input_data: list[str],
    submods: list[Submodule],
    metric_fn: Callable[[Envoy], t.Tensor],
    metric_kwargs: dict = dict(),
    batch_size: int = 32,
    tracer: nnsight.intervention.base.InterleavingTracer = None
) -> tuple[t.Tensor, dict[Submodule, t.Tensor]]:
  """
  Get the activations for the model.
  """
  acts = dict()
  with t.no_grad(), model.generate(input_data, gen_config):
    for submod in model.submodules:
      acts[submod] = submod.get_activation().save()
    metric = metric_fn(model, **metric_kwargs).save()

  return metric, {k: v.value for k, v in acts.items()}


def _single_patch_run(
    model: nnsight.LanguageModel,
    input_data: list[str],
    rep_acts: dict[Submodule, t.Tensor],
    patches: list[tuple],
    metric_fn: Callable,
    metric_kwargs: dict = dict(),
) -> t.Tensor:
  """
  Run the model with different activations once.

  Args:
      model: The model to run the trace on
  Returns:
      Metric value for the model with the corrupted activations
  """
  with t.no_grad(), model.generate(input_data, gen_config):

    for submod, idx in patches:
      original = submod.get_activation()
      # This step requires the same shape or boardcastable.
      original[..., idx] = rep_acts[submod][..., idx]

    metric = metric_fn(model, **metric_kwargs).sum(dim=0).save()

  return metric


def pe_exact(
    model: nnsight.LanguageModel,
    clean_input: list[str],
    clean_acts: dict[Submodule, t.Tensor],
    corrupt_acts: dict[Submodule, t.Tensor],
    clean_metric: t.Tensor,
    corrupt_metric: t.Tensor,
    submods: list[Submodule],
    metric_fn: Callable[[Envoy], t.Tensor],
    metric_kwargs: dict = dict(),
    batch_size: int = 32,
) -> dict[Submodule, t.Tensor]:
  """
  Calculate the indirect effect of all neurons.
  """
  effects = dict()
  for submod in tqdm(submods):
    clean_act = clean_acts[submod]
    corrupt_act = corrupt_acts[submod]
    # For now we only support aggregated activations and metrics.
    assert len(clean_act.shape) == 1 and clean_act.numel() == 1
    assert len(corrupt_act.shape) == 1 and corrupt_act.numel() == 1

    submod_ie = t.zeros_like(clean_act)
    # Loop over indices of all neurons that are different.
    indices = t.nonzero(clean_act - corrupt_act)
    for i in range(indices):
      i = tuple(i)
      patches = [(submod, i)]
      # rep_metric = f(y | x, do(a = a'))
      rep_metric = _single_patch_run(
          model, clean_input, corrupt_acts, patches, metric_fn, metric_kwargs
      )

      # clean_metric = f(y | x), corrupt_metric = f(y | x')
      # ie_normalized = (clean - rep) / (clean - corrupt)
      ie = clean_metric - rep_metric
      submod_ie[i] = ie / (clean_metric - corrupt_metric)
    effects[submod] = submod_ie

  return effects
