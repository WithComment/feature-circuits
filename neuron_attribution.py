import nnsight.tracing
import nnsight.tracing.contexts
from sentence_transformers import SentenceTransformer
import torch as t
import nnsight
from nnsight.intervention import Envoy
from tqdm import tqdm
from typing import Callable

from loading_utils import Submodule
from prompts.util import gen_response, get_max_token_length

ST = SentenceTransformer('all-MiniLM-L6-v2')


def get_embds_of_response(
    input_data: list[str],
    model: nnsight.LanguageModel = None,
    gen_config: dict = dict(),
) -> t.Tensor:
  '''
  If `model` is None, get embeddings of the `input_data` in the sentence
  transformer space.
  If `model` is not None, get embeddings of the generated response of the
  `input_data` in the sentence transformer space.
  '''
  if model is not None:
    input_data = gen_response(model, input_data, gen_config)
  return ST.encode(input_data)


def get_metric(
    model: nnsight.LanguageModel,
    input_data: list[str],
    metric_fn: Callable[[Envoy], t.Tensor],
    metric_kwargs: dict = dict(),
    gen_config: dict = dict(),
    generate: bool = True,
) -> t.Tensor:
  '''
  Get the metric value for the given input data.
  Returns a tensor with the metric value.
  '''
  if generate:
    with t.no_grad(), model.generate(input_data, **gen_config):
      prompt_len = model.gpt_neox.input.shape[1].save()
      model_outputs = model.generator.output.save()
    metric = metric_fn(model, prompt_len, model_outputs, **metric_kwargs)
  
  else:
    with t.no_grad(), model.trace(input_data):
      metric = metric_fn(model, **metric_kwargs)
  return metric


def get_activations(
    model: nnsight.LanguageModel,
    input_data: list[str],
    submods: list[Submodule],
    gen_config: dict = dict(),
    sample_aggre: str = 'mean',
    get_at: str = 'generated',
    pos_aggre: str = 'mean',
) -> dict[Submodule, t.Tensor]:
  '''
  Get activations of the `submods` in the `model` at prompt positions. 
  The activations have shape (N, L, ...). N is the number of examples, 
  L is the length of the prompt,
  and ... is the shape of the activation of the submodule. 

  - If `sample_aggre == 'mean'`, the activations are averaged over the samples.
  - If `get_at == 'last'`, the activations are taken at the last token of the prompt.
  - If `get_at == 'generated'`, the activations are averaged over all generated tokens.
  '''
  acts = dict()
  layers = model.gpt_neox.layers
  with t.no_grad(), model.generate(input_data, **gen_config):
    
  
    if get_at == 'generated':
      for submod in submods:
        acts[submod] = nnsight.list().save()
      layers.all()
      for submod in submods:
        acts[submod].append(submod.get_activation()[:, -1, ...])
        
    elif get_at == 'last':
      for submod in submods:
        acts[submod] = submod.get_activation()[:, -1, ...].save()
  
  for submod in acts.keys():
    if pos_aggre == 'mean':
      acts[submod] = t.stack(acts[submod].value).mean(dim=0)
    if sample_aggre == 'mean':
      acts[submod] = acts[submod].mean(dim=0, keepdim=True)
  
  return acts


def patch_run(
    model: nnsight.LanguageModel,
    input_data: list[str],
    prompt_len: int,
    masks: dict[Submodule, t.Tensor],
    patches: dict[Submodule, t.Tensor],
    metric_fn: Callable,
    metric_kwargs: dict = dict(),
    patch_method: str = 'replace',
    patch_at: str = 'generated',
    gen_config: dict = dict(),
) -> t.Tensor:
  """
  Run the model with patched activations.
  For each (submodule, patch_act) in `patches`, the activation of the submodule
  `patch_act` and the model is run.
  - If patch_pos is 'end', the patch is applied only at the end of the prompt.
  - If patch_pos is 'last', the patch is applied at the last token during each iteration.
  - If patch_pos is 'generated', the patch is applied at all generated tokens.
  """
  pos = prompt_len - 1
  with t.no_grad(), model.generate(input_data, **gen_config, pad_token_id=model.tokenizer.eos_token_id):
    model.gpt_neox.all()
    for submod, mask in masks.items():
      patch_act = patches[submod]
      mod_out = submod.get_activation()
      if patch_method == 'add':
        raise NotImplementedError

      match patch_at:
        case 'last':
          clean_act = mod_out[:, -1, ...]
          mod_out[:, -1, ...] = t.where(mask, patch_act, clean_act)
        case 'generated':
          clean_act = mod_out[:, pos:, ...]
          mod_out[:, pos:, ...] = t.where(mask, patch_act, clean_act)

    model_outputs = model.generator.output.save()

  metric = metric_fn(model, prompt_len, model_outputs, **metric_kwargs)
  return metric


def pe_exact(
    model: nnsight.LanguageModel,
    clean_input: list[str],
    clean_acts: dict[Submodule, t.Tensor],
    patch_acts: dict[Submodule, t.Tensor],
    submods: list[Submodule],
    metric_fn: Callable[[Envoy], t.Tensor],
    metric_kwargs: dict = dict(),
    patch_method: str = 'replace',
    patch_pos: str = 'generated',
    gen_config: dict = dict(),
) -> dict[Submodule, t.Tensor]:
  """
  Calculate the indirect effect of all neurons.
  Return a dictionary of submodules and their indirect effects.
  The shape of the indirect effects are the same as the activations.
  """
  prompt_len = get_max_token_length(model, clean_input)
  itv_metrics = dict()
  for submod in tqdm(submods):
    clean_act = clean_acts[submod]
    patch_act = patch_acts[submod]
    submod_ie = t.zeros(*patch_act.shape[1:], len(clean_input)).to(model.device)

    for idx in t.nonzero(t.any(t.ne(clean_act, patch_act), dim=0)):
      mask = t.zeros(*clean_act.shape[1:], patch_act.shape[0], dtype=t.bool)
      mask[idx] = True
      mask = mask.permute(-1, *range(0, mask.ndim - 1))
      submod_ie[idx] = patch_run(
          model,
          clean_input,
          prompt_len,
          {submod: mask},
          patch_acts,
          metric_fn,
          metric_kwargs,
          patch_method,
          patch_pos,
          gen_config,
      )
    itv_metrics[submod] = submod_ie
  return itv_metrics


def get_rel_effect(
    clean_metric: t.Tensor,
    patch_metric: t.Tensor,
    itv_metric: dict[Submodule, t.Tensor],
) -> dict[Submodule, t.Tensor]:
  clean_metric = clean_metric.detach().cpu()
  patch_metric = patch_metric.detach().cpu()
  total = (patch_metric - clean_metric)
  total[total == 0] = total.mean()
  rel_effect = dict()
  max_ = -t.inf
  min_ = t.inf
  positive = 0
  total_params = 0
  for submod, metric in itv_metric.items():
    rel_effect[submod] = ((metric.detach().cpu() - clean_metric) / total)
    max_ = max(max_, rel_effect[submod].max())
    min_ = min(min_, rel_effect[submod].min())
    positive += t.sum(rel_effect[submod] > 0)
    total_params += rel_effect[submod].numel()

  print(f"Total effect: {total}")
  print(f"Max: {max_:.4f}, Min: {min_:.4f}, Positive rate: {positive / total_params:.4f}")
  return rel_effect


def get_circuit_mask(
    effects: dict[Submodule, t.Tensor],
    threshold: float,
    absolute: bool = False
) -> dict[Submodule, t.Tensor]:
  """
  Filter out the neurons with indirect effects below the threshold.
  """
  included = 0
  total = 0
  result = dict()
  for k, v in effects.items():
    v = v.mean(dim=-1).unsqueeze(0)
    total += v.numel()
    if absolute:
      v = t.abs(v)
    included += t.sum(v > threshold)
    result[k] = v > threshold
  
  print(f"Inclusion rate: {included}/{total} ({included / total:.4f})")
  return result


def eval_circuit(
    model: nnsight.LanguageModel,
    clean_prompts: list[str],
    patch_prompts: list[str],
    circuit_masks: dict[Submodule, t.Tensor],
    circuit_acts: dict[Submodule, t.Tensor] | None,
    metric_fn: Callable[[Envoy], t.Tensor],
    metric_kwargs: dict = dict(),
    gen_config: dict = dict(),
    patch_method: str = 'replace',
    patch_pos: str = 'generated',
    sample_aggre: str = 'mean',
) -> dict[Submodule, t.Tensor]:
  """
  Evaluate the faithfulness of the circuit.
  The faithfulness is defined as m(C) - m()
  """
  clean_metric = get_metric(
      model, clean_prompts, metric_fn, metric_kwargs, gen_config)
  patch_metric = get_metric(
      model, patch_prompts, metric_fn, metric_kwargs, gen_config)
  
  if circuit_acts is None:
    circuit_acts = get_activations(
        model, patch_prompts, list(circuit_masks), gen_config, sample_aggre=sample_aggre)
    

  itv_metric = patch_run(
      model, clean_prompts, get_max_token_length(model, clean_prompts),
      circuit_masks, circuit_acts, metric_fn, metric_kwargs, patch_method, patch_pos, gen_config)
  
  print(patch_metric - clean_metric)
  return (itv_metric - clean_metric) / (patch_metric - clean_metric)
