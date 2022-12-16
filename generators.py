import jax
import jax.numpy as jnp
import haiku as hk

from generators.conditional_data import ConditionalData
from generators.conditional_gen import ConditionalGenerator
from generators.conditional_residual_gen import ConditionalResidualGenerator

import configlib

parser = configlib.add_parser("Genrator config")
parser.add_argument("--generator_model", default="conditional_gen", type=str, metavar="GENERATOR_NAME",
        help="The classifier architecture to train (xy_conditional_gen, condistional_gen, conditional_residual_gen, conditional_data).")
parser.add_argument("--im_dim", type=int, metavar="DIM", default=32,
        help="The dimension of the (squared) image.")
parser.add_argument("--im_chan", type=int, default=3,
        help="The number of channels in the images.")
parser.add_argument("--z_dim", type=int, default=64,
        help="The dimension of the randomness used by the generator for each image generated.")
parser.add_argument("--conditional_label_encoding", type=str, default="one_hot",
        help="The label/class encoding scheme to use in conditional generators (embedding, one_hot).")
parser.add_argument("--images_per_class", type=int, default=1,
        help="Number of images per class to learn when learning the data directly.")
parser.add_argument("--beta1_g", default=0.9, type=float,
        help="Generator optimizer beta1.")
parser.add_argument("--beta2_g", default=0.999, type=float,
        help="Generator optimizer beta2.")
parser.add_argument("--min_loss_n", type=int, default=1,
        help="Minimum loss over n draws of noise when training the generator.")

parser.add_argument("--batch_conditioning", default=False, action='store_true',
        help="Generate data conditioned on the batch (generated data can specialize to each minibatch of classifier training).")
parser.add_argument("--batch_embedding_size", type=int, default=10,
        help="Learned embedding size of the btach number encoding.")

def get_forward_gen(c: configlib.Config, GeneratorModule, n_batches=None):
    if c.generator_model == "conditional_data":
        def get_noise(key, batch_size):
            return jax.random.randint(key, (batch_size,), 0, c.images_per_class)
    else:
        def get_noise(key, batch_size):
            return jax.random.normal(key, (batch_size, c.z_dim))

    def get_labels(key, batch_size):
        return jax.random.randint(key, (batch_size,), 0, c.n_classes)

    if c.batch_conditioning:
        def get_batches(key, batch_size):
            return jax.random.randint(key, (batch_size,), 0, n_batches)

    def fwd(batch_size, is_training, noise=None, labels=None, batches=None):
        if noise is None:
            noise  = get_noise(hk.next_rng_key(), batch_size)
        if labels is None:
            labels = get_labels(hk.next_rng_key(), batch_size)
        if c.batch_conditioning and batches is None:
            batches = get_batches(hk.next_rng_key(), batch_size)

        gen = GeneratorModule(c, n_batches)
        images = gen(noise, labels, batches, is_training)

        return images, labels

    return fwd

def get_generator(c: configlib.Config, n_batches=None):
    if c.generator_model == "conditional_residual_gen":
        return get_forward_gen(c, ConditionalResidualGenerator, n_batches)
    elif c.generator_model == "conditional_gen":
        return get_forward_gen(c, ConditionalGenerator, n_batches)
    elif c.generator_model == "conditional_data":
        return get_forward_gen(c, ConditionalData, n_batches)
    else:
        # TODO: add back label gen
        raise NotImplementedError
