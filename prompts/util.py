import os
import random
import nnsight
import torch as t

import json


def get_max_token_length(
    model: nnsight.LanguageModel,
    input_data: list[str]
) -> int:
  """
  Calculate the maximum number of tokens in a batch after tokenization.

  Args:
      model: The language model with a tokenizer
      input_data: List of input strings to be tokenized

  Returns:
      The maximum token length across all inputs in the batch
  """
  # Tokenize all inputs in batch
  encodings = model.tokenizer(input_data, padding=False, truncation=False)

  # Find the maximum length
  max_length = max(len(tokens) for tokens in encodings.input_ids)

  return max_length


def gen_response(
    model: nnsight.LanguageModel,
    prompts: list,
    gen_config: dict,
) -> list[str]:
  with t.no_grad(), model.generate(prompts, **gen_config):
    prompt_len = model.gpt_neox.input.shape[1].save()
    out = model.generator.output.save()

  sentences = model.tokenizer.batch_decode(out[:, prompt_len:])

  return sentences


def make_json_dataset(
    clean_prompts_path: str,
    patched_prompts_path: str,
    out_file_path: str,
    model: nnsight.LanguageModel,
    gen_config: dict,
    force: bool = False,
) -> None:

  if os.path.exists(out_file_path) and not force:
    print(f"Dataset already exists at {out_file_path}.")
    return

  clean_prompts = list()
  patch_prompts = list()

  with open(clean_prompts_path, 'r') as f:
    clean_prompts = f.readlines()

  with open(patched_prompts_path, 'r') as f:
    patch_prompts = f.readlines()

  generated = gen_response(model, patch_prompts, gen_config)

  d = [
      {'clean': c, 'patch': p, 'patch_response': r}
      for c, p, r in zip(clean_prompts, patch_prompts, generated)
  ]

  with open(out_file_path, 'w') as f:
    json.dump(d, f, indent=2)


def load_and_split_dataset(
    file_path: str,
    num_train: int = 150,
    num_val: int = 50,
    shuffle: bool = True
) -> tuple[tuple[list[str], list[str], list[str]], tuple[list[str], list[str], list[str]]]:
  """
  Load a JSON dataset file and split it into train and validation sets.
  Each dataset is further split into clean, patch, and target lists.
  
  Args:
      file_path: Path to the JSON dataset file
      num_train: Number of samples for training
      num_val: Number of samples for validation
      shuffle: Whether to shuffle the dataset before splitting
      
  Returns:
      A tuple of two tuples: (train_data, val_data)
      Each inner tuple contains: (clean, patch, target)
  """
  with open(file_path, 'r') as f:
    samples = json.load(f)
  
  if shuffle:
    random.shuffle(samples)
  
  train_samples = samples[:num_train]
  val_samples = samples[num_train:num_train+num_val]
  
  # Split training data
  train_clean = [sample['clean'] for sample in train_samples]
  train_patch = [sample['patch'] for sample in train_samples]
  train_target = [sample['patch_response'] for sample in train_samples]
  
  # Split validation data
  val_clean = [sample['clean'] for sample in val_samples]
  val_patch = [sample['patch'] for sample in val_samples]
  val_target = [sample['patch_response'] for sample in val_samples]
  
  return (train_clean, train_patch, train_target), (val_clean, val_patch, val_target)


def load_prefix_answer_dataset(
    file_path: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
  """
  Load a dataset from a JSON file where each line has the format:
  {
      "clean_prefix": "The friends that the dancer visits", 
      "patch_prefix": "The friend that the dancer visits", 
      "clean_answer": " go", 
      "patch_answer": " goes", 
      "case": "plural_singular"
  }

  Returns:
      A tuple of four lists: (clean_prefixes, clean_answers, patch_prefixes, patch_answers)
  """
  clean_prefixes = []
  clean_answers = []
  patch_prefixes = []
  patch_answers = []

  with open(file_path, 'r') as f:
    for line in f:
      data = json.loads(line.strip())
      clean_prefixes.append(data['clean_prefix'])
      clean_answers.append(data['clean_answer'])
      patch_prefixes.append(data['patch_prefix'])
      patch_answers.append(data['patch_answer'])

  return clean_prefixes, clean_answers, patch_prefixes, patch_answers
