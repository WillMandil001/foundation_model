from absl import app, flags, logging
import flax
import jax
import optax
import tensorflow as tf
import numpy as np
import tqdm
import wandb

from octo.data.dataset import make_single_dataset
from octo.data.utils.data_utils import NormalizationType
from octo.model.components.action_heads import L1ActionHead
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import ( freeze_weights, merge_params, process_text, TrainState )

batch_size = 1
horizon = 2



# set up the model:
pretrained_path = "hf://rail-berkeley/octo-base"
initialize_compilation_cache()
pretrained_model = OctoModel.load_pretrained(pretrained_path)
print(pretrained_model.get_pretty_spec())
config = pretrained_model.config
del config["model"]["observation_tokenizers"]["wrist"]
text_processor = pretrained_model.text_processor
text = "pick up the ball and throw it to the big red dog, then slap the ball with the bat."
batch = {"task": {"language_instruction": [text.encode('utf-8')]},
        "observation": {"image_primary": np.random.uniform(0, 256, size=(batch_size, horizon, 256, 256, 3)).astype(np.int8),
                        # "proprio": np.random.uniform(-2.0, 2.0, size=(batch_size, horizon, action_dim)).astype(np.float32),
                        "pad_mask": np.ones((batch_size, horizon)).astype(np.float32)}}
example_batch = process_text(batch, text_processor)
# # config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(LowdimObsTokenizer, n_bins=256, bin_type="normal", low=-2.0, high=2.0, obs_keys=["proprio"])
config["model"]["heads"]["action"] = ModuleSpec.create(L1ActionHead, pred_horizon=horizon, action_dim=7, readout_key="readout_action")
model = OctoModel.from_config(config, example_batch, text_processor, verbose=True)
merged_params = merge_params(model.params, pretrained_model.params)
model = model.replace(params=merged_params)
print(model.get_pretty_spec())
del pretrained_model



# def process_batch(batch):
#     batch = process_text(batch, text_processor)
#     return batch

# pretrained_path = "hf://rail-berkeley/octo-base"

# initialize_compilation_cache()

# tf.config.set_visible_devices([], "GPU")

# pretrained_model = OctoModel.load_pretrained(pretrained_path)
# print(pretrained_model.get_pretty_spec())

# config = pretrained_model.config
# del config["model"]["observation_tokenizers"]["wrist"]

# text_processor = pretrained_model.text_processor

# text = "pick up the ball and throw it to the big red dog, then slap the ball with the bat."
# batch = {
#         "task": {"language_instruction": [text.encode('utf-8')]},
#         "observation": {
#                         "proprio": np.random.uniform(-2.0, 2.0, size=(batch_size, horizon, 14)).astype(np.float32),
#                         "image": np.random.uniform(0, 256, size=(batch_size, horizon, 256, 256, 3)).astype(np.int8),
#                         "pad_mask": np.ones((batch_size, horizon)).astype(np.float32)
#                         }
#     }
# example_batch = process_batch(batch)

# config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(LowdimObsTokenizer, n_bins=256, bin_type="normal", low=-2.0, high=2.0, obs_keys=["proprio"])
# config["model"]["heads"]["action"] = ModuleSpec.create(L1ActionHead, pred_horizon=horizon, action_dim=14, readout_key="readout_action")
# model = OctoModel.from_config(config, example_batch, text_processor, verbose=True)

# merged_params = merge_params(model.params, pretrained_model.params)
# model = model.replace(params=merged_params)
# del pretrained_model


# # run the model once and get an output
# # language_instruction = batch["task"]["language_instruction"]
# task = model.create_tasks(texts=["please pick up the ball and place on the red mat"])

# obs = {
#         "proprio": np.random.uniform(-2.0, 2.0, size=(horizon, 14)).astype(np.float32),
#         "image": np.random.uniform(0, 256, size=(horizon, 256, 256, 3)).astype(np.int8),
#         "pad_mask": np.ones((horizon)).astype(np.float32)
#     }

# actions = model.sample_actions(jax.tree_map(lambda x: x[None], obs), task, rng=jax.random.PRNGKey(0))
# actions = actions[0]
