#!/usr/bin/env python3
# filepath: /h/xiaowenz/494/feature_circuits/run_experiment.py

import os
import pickle
import torch as t
import argparse
import matplotlib.pyplot as plt
from nnsight import LanguageModel
from transformers import set_seed

from dictionary_loading_utils import load_saes_and_submodules
from prompts.util import make_json_dataset, load_and_split_dataset
from neuron_attribution import (
    get_embds_of_response, get_activations, get_metric,
    pe_exact, get_rel_effect, get_circuit_mask, eval_circuit
)
from metrics import target_similarity


def parse_args():
  parser = argparse.ArgumentParser(
      description='Run feature circuits experiment')

  # Model parameters
  parser.add_argument('--model', type=str, default="EleutherAI/pythia-70m-deduped",
                      help='Model name')
  parser.add_argument('--device', type=str, default="cuda:0",
                      help='Device to run the model on')
  parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
  parser.add_argument('--dtype', type=str, default="float32",
                      choices=["float16", "float32", "bfloat16"],
                      help='Model precision')

  # Data parameters
  parser.add_argument('--category', type=str, default="concise",
                      help='Category of prompts to use')
  parser.add_argument('--num_train', type=int, default=32,
                      help='Number of training examples')

  # Generation config
  parser.add_argument('--max_new_tokens', type=int, default=30,
                      help='Maximum number of new tokens to generate')
  parser.add_argument('--do_sample', action='store_true',
                      help='Whether to use sampling instead of greedy decoding')
  parser.add_argument('--repetition_penalty', type=float, default=1.2,
                      help='Repetition penalty')

  # Experiment parameters
  parser.add_argument('--sample_aggre', type=str, default="pair",
                      choices=["mean", "pair"],
                      help='Sample aggregation method')
  parser.add_argument('--get_at', type=str, default="generated",
                      choices=["last", "generated"],
                      help='Position to get activations at')
  parser.add_argument('--pos_aggre', type=str, default="mean",
                      choices=["mean", "end"],
                      help='Position aggregation method')
  parser.add_argument('--patch_method', type=str, default="replace",
                      choices=["replace", "add"],
                      help='Method to patch activations')
  parser.add_argument('--patch_at', type=str, default="generated",
                      choices=["last", "generated"],
                      help='Position to patch activations at')
  parser.add_argument('--threshold', type=float, default=1.0,
                      help='Threshold for circuit mask')

  # Output parameters
  parser.add_argument('--save_dir', type=str, default="effects",
                      help='Directory to save effects')
  parser.add_argument('--fig_dir', type=str, default="figures",
                      help='Directory to save figures')
  parser.add_argument('--skip_gen_dataset', action='store_true',
                      help='Skip dataset generation')

  return parser.parse_args()


def main():
  args = parse_args()

  # Set seed for reproducibility
  set_seed(args.seed)

  # Set device and dtype
  device = t.device(args.device)
  if args.dtype == "float32":
    dtype = t.float32
  elif args.dtype == "float16":
    dtype = t.float16
  elif args.dtype == "bfloat16":
    dtype = t.bfloat16

  # Set up model
  print(f"Loading model {args.model}...")
  model = LanguageModel(
      args.model,
      device_map=device,
      dispatch=True,
      torch_dtype=dtype,
  )

  # Set up generation config
  gen_config = {
      'max_new_tokens': args.max_new_tokens,
      'do_sample': args.do_sample,
      'repetition_penalty': args.repetition_penalty,
  }

  # Set up paths
  prompts_dir = 'prompts'
  cp_path = os.path.join(prompts_dir, 'prompts.txt')
  pp_path = os.path.join(prompts_dir, f'{args.category}_prompts.txt')
  out_path = os.path.join(prompts_dir, f'{args.category}.json')

  # Create output directories if they don't exist
  os.makedirs(args.save_dir, exist_ok=True)
  os.makedirs(args.fig_dir, exist_ok=True)

  # Generate dataset if needed
  if not args.skip_gen_dataset:
    print("Generating dataset...")
    make_json_dataset(cp_path, pp_path, out_path, model, gen_config)

  # Load submodules
  print("Loading submodules...")
  submods, SAEs = load_saes_and_submodules(
      model,
      separate_by_type=True,
      include_embed=True,
      device=device,
      dtype=dtype,
      neurons=True,
  )

  # Load dataset
  print("Loading dataset...")
  ds, val_ds = load_and_split_dataset(out_path, args.num_train)
  train_clean, train_patch, train_target = ds
  train_target_embds = get_embds_of_response(train_target)

  val_clean, val_patch, val_target = val_ds
  val_target_embds = get_embds_of_response(val_target)

  # Define base name for saving
  save_base = f'{args.model.split("/")[-1]}_{args.category}_{args.sample_aggre}_{args.get_at}_{args.pos_aggre}_{args.patch_method}_{args.patch_at}'
  save_base = save_base.replace('-', '_')

  # Clean run
  print("Getting clean activations...")
  metric_kwargs = dict(target_embds=train_target_embds)
  submods_flat = submods.resids

  clean_acts = get_activations(
      model,
      train_clean,
      submods_flat,
      gen_config=gen_config,
      sample_aggre=args.sample_aggre,
      get_at=args.get_at,
      pos_aggre=args.pos_aggre,
  )

  clean_metric = get_metric(
      model,
      train_clean,
      target_similarity,
      metric_kwargs,
      gen_config
  )

  # Patched run
  print("Getting patched activations...")
  patch_acts = get_activations(
      model,
      train_patch,
      submods_flat,
      gen_config=gen_config,
      sample_aggre=args.sample_aggre,
      get_at=args.get_at,
      pos_aggre=args.pos_aggre,
  )

  patch_metric = get_metric(
      model,
      train_patch,
      target_similarity,
      metric_kwargs,
      gen_config
  )

  # Calculate intervention metric
  print("Calculating intervention metrics...")
  itv_metric = pe_exact(
      model,
      train_clean,
      clean_acts,
      patch_acts,
      submods_flat,
      metric_fn=target_similarity,
      metric_kwargs=metric_kwargs,
      gen_config=gen_config,
      patch_method=args.patch_method,
      patch_pos=args.patch_at,
  )

  # Calculate relative effect
  print("Calculating relative effect...")
  rel_effect = get_rel_effect(clean_metric, patch_metric, itv_metric)
  rel_effect_mat = t.stack(tuple(rel_effect.values())
                           ).transpose(1, 0).nanmean(dim=-1)

  # Save results
  print(f"Saving results to {args.save_dir}/{save_base}.pkl")
  with open(f'{args.save_dir}/{save_base}.pkl', 'wb') as f:
    pickle.dump((itv_metric, clean_metric, patch_metric, rel_effect_mat), f)

  # Plot results
  print(f"Saving figure to {args.fig_dir}/{save_base}.png")
  plt.figure(figsize=(10, 50))
  plt.matshow(rel_effect_mat, fignum=1, aspect='auto')
  plt.colorbar()
  plt.title(
      f"Relative effect for {args.model.split('/')[-1]}, {args.category}")
  plt.savefig(f'{args.fig_dir}/{save_base}.png')

  # Get circuit mask
  print("Getting circuit mask...")
  masks = get_circuit_mask(rel_effect, args.threshold)

  # Evaluate circuit
  print("Evaluating circuit...")
  metric_kwargs = dict(target_embds=val_target_embds)
  circuit_perf = eval_circuit(
      model,
      val_clean,
      val_patch,
      masks,
      {k: v.mean(dim=0, keepdim=True) for k, v in patch_acts.items()},
      target_similarity,
      metric_kwargs,
      gen_config,
      patch_method=args.patch_method,
      patch_pos=args.patch_at,
  ).mean()

  print(f"Circuit performance: {circuit_perf}")

if __name__ == "__main__":
  main()
