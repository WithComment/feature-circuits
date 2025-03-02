import nnsight
import torch as t


def count_generated_tokens(model: nnsight.LanguageModel, prompt: str, max_tokens: int = 500) -> int:
  """
  Generates text from a prompt and returns the number of tokens generated.
  Continues generating until either a BoS token is encountered or max_tokens is reached.

  Args:
      model: The language model to use for generation
      prompt: Input prompt string
      max_tokens: Maximum number of new tokens to generate (default: 100)

  Returns:
      int: Number of tokens generated (excluding prompt tokens)
  """

  # Convert prompt to tensor if needed
  if isinstance(prompt, str):
    prompt = prompt.strip()
    encoded_prompt = prompt.encode('utf-8', errors='ignore').decode('utf-8')
    print(encoded_prompt)
    prompt_tensor = model.tokenizer(
      prompt, return_tensors="pt", truncation=True, max_length=2048)["input_ids"]
  else:
    prompt_tensor = prompt

  prompt_length = prompt_tensor.shape[-1]

  # Get the BoS token ID - usually it's the same as the model's bos_token_id
  bos_token_id = model.config.bos_token_id

  # Generate text
  with model.generate(prompt_tensor, max_new_tokens=max_tokens) as generator:
    out = model.generator.output.save()

  generated_tokens = total_tokens - prompt_length

  assert generated_tokens > 1
  return generated_tokens


def length_metric(
    model: nnsight.LanguageModel,
    clean_prompts: t.Tensor,
    patch_prompts: t.Tensor,
):
  """
  Calculates the difference in average response length between clean and patched prompts.

  Args:
      model: The language model to use for generating responses
      clean_prompts: Tensor of clean prompts
      patch_prompts: Tensor of patched prompts

  Returns:
      float: The difference between the average length of responses to clean prompts and patched prompts (clean - patched)
  """
  clean_lengths = list()
  for i in range(len(clean_prompts)):
    clean_lengths.append(count_generated_tokens(model, clean_prompts[i]))
  
  patched_lengths = list()
  for i in range(len(patch_prompts)):
    patched_lengths.append(count_generated_tokens(model, patch_prompts[i]))
  clean_lengths = t.tensor(clean_lengths).log()
  patched_lengths = t.tensor(patched_lengths).log()

  return clean_lengths - patched_lengths
