import nnsight
import torch as t
from sentence_transformers import SentenceTransformer


def length_metric(
    model: nnsight.LanguageModel,
):
  """
  Calculates the difference in average response length between clean and patched prompts.

  Args:
      model: The language model to use for generating responses
      patched: Tensor of clean prompts
      prompts: Tensor of patched prompts

  Returns:
      float: The difference between the average length of responses to clean prompts and patched prompts (clean - patched)
  """
  return model.output


def eos_metric(
    model: nnsight.LanguageModel,
) -> t.Tensor:
  """Return the logits of the end of sequence token."""
  return -model.output.logits[:, -1, model.tokenizer.bos_token_id]
