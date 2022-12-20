import os
import pickle
from torch.utils.data import Dataset

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import utils
from generators.generators import get_generator

'''
Custom dataset used to load gradients and params from trained
classifier
'''


class GradDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.steps = self.get_steps()

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        grad_path = self.root_dir + "/g_" + str(idx) + ".pickle"

        with open(grad_path, 'rb') as f:
            g_struct = pickle.load(f)

        g_struct["batch_id"] = idx

        return g_struct

    def get_steps(self):
        steps_path = self.root_dir + "/steps.pickle"
        with open(steps_path, 'rb') as f:
            steps = pickle.load(f)
            return steps["steps"]

class GradSubset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.valid_steps = self.get_valid_steps()

    def get_valid_steps(self):
        steps_path = self.root_dir + "/steps.pickle"
        with open(steps_path, 'rb') as f:
            steps = pickle.load(f)
            steps = steps["steps"]

        valid_steps = []
        for idx in range(steps):
            grad_path = self.root_dir + "/metadata_" + str(idx) + ".pickle"
            with open(grad_path, 'rb') as f:
                meta_struct = pickle.load(f)
                if meta_struct["grads_bellow_size"] == 0 and meta_struct["grads_above_size"] == 0:
                    valid_steps.append(idx)

        return valid_steps

    def __len__(self):
        return len(self.valid_steps)

    def __getitem__(self, idx):
        idx = self.valid_steps[idx]
        grad_path = self.root_dir + "/g_" + str(idx) + ".pickle"

        with open(grad_path, 'rb') as f:
            g_struct = pickle.load(f)

        return g_struct


'''
Custome dataset from trained generator
'''


class SynDataset(Dataset):
    """Custom dataset from trained generator"""

    def __init__(self, config, ds_size, gen_path=None, transform=None):
        """
        Args:
            config (dict): info about the training parameters
            ds_size (int): size of the dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.config = config

        self.g_prng_seq = hk.PRNGSequence(0)
        self.model_g = hk.transform_with_state(get_generator(config))
        self.z_dim = config["z_dim"]
        self.n_classes = config["c_out"]

        # Get trained generator parameters
        if gen_path is None:
            self.gen_path = os.path.join(config["run_dir"], "trained_gen.pickle")
        else:
            self.gen_path = gen_path

        with open(self.gen_path, 'rb') as f:
            gen_struct = pickle.load(open(self.gen_path, 'rb'))
        self.ds_size = ds_size

        # Generate samples
        (self.samples, labels), _ = self.model_g.apply(gen_struct["params"],
                                   gen_struct["state"],
                                   next(self.g_prng_seq),
                                   batch_size = self.ds_size,
                                   is_training=False)

        self.labels = jnp.expand_dims(labels, axis=1)

    def __len__(self):
        return self.ds_size

    def __getitem__(self, idx):
        x = self.samples[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y
