from metrics import eos_metric
from attribution import get_activations
import os
from data_loading_utils import load_examples_rct
from dictionary_loading_utils import load_saes_and_submodules

from nnsight import LanguageModel
import torch as t

device = t.device('cuda:0')
dtype = t.float32

model = LanguageModel(
    'EleutherAI/pythia-70m-deduped',
    device_map=device,
    dispatch=True,
    torch_dtype=dtype,
)

category = 'concise'
num_examples = 100
train_clean, train_patch = load_examples_rct(
    os.path.join('prompts', category), 50)
save_base = f'pythia-70m-deduped_prompt_dir_{category}_n{num_examples}'

submods, SAEs = load_saes_and_submodules(
    model,
    separate_by_type=False,
    include_embed=True,
    device=device,
    dtype=dtype,
)


metric_fn = eos_metric

clean_acts, clean_metric = get_activations(
    model,
    train_clean,
    submods,
    SAEs,
    calc_metric=True,
    metric_fn=metric_fn,
)
