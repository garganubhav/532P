import os
import pickle
import glob
from tqdm import tqdm
import jax
from jax import jit
from functools import partial
import jax.random
import haiku as hk
import time
import numpy as np
import jax.numpy as jnp
from jax.experimental.host_callback import call
import optax
import losses
import torch
import utils
import dp_accountant
from training.offline.fid import calculate_fid_given_paths
from data_utils.datasets import get_dataset
from data_utils import dataset
from data_utils.datasets import get_dataset
from data_utils.dataloader import GradLoader, get_generated_dataset
from classifiers.classifiers import get_classifier
from generators.generators import get_generator
from training.optimizers import get_classifier_optimizer
from training.utils import get_accuracy, clip_grads, grads_norm_fn, noise_grads, get_compute_grads, generate_and_save_images, update_model
from training.loss_functions import get_classifier_training_loss_fn
from generators.evaluation import get_synthetic_accuracy
import configlib
'''
Returns a function that computes the classifier loss on the fake data.
'''

def get_c_loss_fake(model_c):
    @jax.jit
    def c_loss_fake(params, fake_img, fake_img_label):
        pred = model_c.apply(params, fake_img)
        loss = losses.soft_x_ent(pred, jnp.expand_dims(fake_img_label, axis=0))
        return loss

    return c_loss_fake


'''
Gets cosine similarity between two gradients
'''
def cos_sim(x, y):
    dot = jnp.sum(x * y, axis=1)
    norm = jnp.linalg.norm(x, axis=1) * jnp.linalg.norm(y, axis=1)
    return dot / norm

def norm_diff(x, y):
    return jnp.linalg.norm(x, axis=1) - jnp.linalg.norm(y, axis=1)

def sum_squared_diff(x, y):
    return jnp.sum(jnp.square(x - y), axis=1)

@jax.jit
def l2_reg(inputs):
    loss_l2 = jnp.linalg.norm(
        jnp.reshape(inputs, (inputs.shape[0], -1)),
        axis=1
    ).mean()
    return loss_l2

@jax.jit
def tv_l1_reg(inputs):
    # https://www.wikiwand.com/en/Total_variation_denoising
    # COMPUTE total variation regularization loss
    diff1 = inputs[:, :, :, :-1] - inputs[:, :, :, 1:]
    diff2 = inputs[:, :, :-1, :] - inputs[:, :, 1:, :]

    diff1 = jnp.absolute(diff1).sum(axis=(1,2,3))
    diff2 = jnp.absolute(diff2).sum(axis=(1,2,3))

    loss_tv_l1 = (diff1 + diff2).mean()
    return loss_tv_l1

@jax.jit
def tv_l2_reg(inputs):
    # https://www.wikiwand.com/en/Total_variation_denoising
    # COMPUTE total variation regularization loss
    diff1 = inputs[:, :, :, :-1] - inputs[:, :, :, 1:]
    diff2 = inputs[:, :, :-1, :] - inputs[:, :, 1:, :]

    diff1 = jnp.linalg.norm(
        jnp.reshape(diff1, (inputs.shape[0], -1)),
        axis=1
    )
    diff2 = jnp.linalg.norm(
        jnp.reshape(diff2, (inputs.shape[0], -1)),
        axis=1
    )

    loss_tv_l2 = (diff1 + diff2).mean()
    return loss_tv_l2

@jax.jit
def fid_reg(images):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    #conf.batch_size = fake.shape[0]
    #train_dataset, test_dataset, train_loader, test_loader = get_dataset(conf)

    #images = [fake, next(iter(train_loader))]
    f_arr =  jnp.asarray(images[0])
    images[0] = torch.from_numpy(f_arr)
    o_arr =  jnp.asarray(images[1])
    images[1] = torch.from_numpy(o_arr)
    print("FAKE=", images[0], "ORIGINAL= ",images[1])
    fid_value = calculate_fid_given_paths(images,
                                          device,
                                          2048,
                                          8)
    print('FID: ', fid_value)
    return fid_value

'''
Returns a function that computes the generator loss from gradients of the classifier.
'''
def get_g_loss(conf, model_g, z_dim, n_classes, im_chan, loss_fn_c, c, b_size_g, b_size_c, min_loss_n, layers=None):
    compute_grads_c = get_compute_grads(
            conf, loss_fn_c, do_clip_grads=not conf.disable_dp, do_noise_grads=False, get_metadata=False
    )
    # We need to handle the keyword arguments for vmap, and put labels on the
    # correct dimension for vmap:
    def clipped_grads_c(params, inputs, labels, clip):
        return compute_grads_c(params, inputs, jnp.expand_dims(labels, axis=1), clip=clip)


    if layers is None:
        layer_selection = lambda x: x
    elif type(layers) == list:
        layer_selection = lambda x: [jax.tree_util.tree_flatten(x)[0][i] for i in layers]
    else:
        layer_selection = lambda x: jax.tree_util.tree_flatten(x)[0][layers]

    @jit
    def _batch_loss(fake, labels, params_c, c_grad_dp):
        vmap_grads_c = jax.vmap(clipped_grads_c, in_axes=(0, 0, 0, None))
        c_grad_fake = vmap_grads_c(params_c, fake, labels, c)
        c_grad_fake = clip_grads(c_grad_fake, c)

        # select layer to train on
        c_grad_dp = layer_selection(c_grad_dp)
        c_grad_fake = layer_selection(c_grad_fake)

        grads_dp_flat = utils.flatten_jax_map_batch(c_grad_dp, b_size_g)
        grads_fake_flat = utils.flatten_jax_map_batch(c_grad_fake, b_size_g)

        conf["batch_size"] = fake.shape[0]
        train_dataset, test_dataset, train_loader, test_loader = get_dataset(conf)

        images = [fake, next(iter(train_loader))]

        loss = cos_sim(grads_dp_flat, grads_fake_flat)
        loss = -(loss - 1)

        if conf.g_tv_l1 is not None:
            loss = loss + conf.g_tv_l1 * tv_l1_reg(fake)

        if conf.g_tv_l2 is not None:
            loss = loss + conf.g_tv_l2 * tv_l2_reg(fake)

        if conf.g_l2_reg is not None:
            loss = loss + conf.g_l2_reg * l2_reg(fake)

        if conf.g_fid_reg is not None:
            loss = loss + conf.g_fid_reg * fid_reg(images)

        #  loss = jax.numpy.arccos(loss)
        #loss = -(loss - 1)  # To have th rescaled cos instead

        # NOTE: (squared) norm difference between grads
        #  loss += jnp.square(norm_diff(grads_dp_flat, grads_fake_flat))

        # NOTE: sum suared error betwenn grads (squared l2 norm of the diff)
        #  loss += sum_squared_diff(grads_dp_flat, grads_fake_flat)

        return loss

    @jax.jit
    def g_loss(params_g, state_g, params_c, c_grad_dp, c_batch_ids, prng_key):
        batch_size = min_loss_n * b_size_g * b_size_c

        # NOTE: This means that we are doing batch norm over the whole
        # generation for each fake data batch, and each gen batch, together. Is
        # this what we want? How do we do it differently?
        # Maybe generate within the inner-most funciton, in the first line of
        # _batch_loss? But then which state do we carry-over? One of the
        # batches and that's it?

        (fake, labels), state_g = model_g.apply(params_g, state_g, prng_key, batch_size, True, batches=c_batch_ids)
        #  labels = jax.nn.one_hot(labels, n_classes)
        fake = jnp.reshape(fake, (min_loss_n, b_size_g, b_size_c, *fake.shape[1:]))
        labels = jnp.reshape(labels, (min_loss_n, b_size_g, b_size_c, *labels.shape[1:]))

        loss = jax.vmap(_batch_loss, in_axes=(0, 0, None, None))(fake, labels, params_c, c_grad_dp)

        loss = jnp.min(loss, axis=0, keepdims=False)

        agg_loss = jnp.mean(loss)
        #  agg_loss = jnp.min(loss)

        return agg_loss, state_g

    return g_loss


'''
Gets gradient update for classifier computed on the fake data from the generator. 
Uses gradients from classifier computed on the real data with DP and on the fake
data from the generator to update the generator.
'''
def get_update_g(model_c, model_g, loss_fn_g, opt_g):
    @jax.jit
    def update_g(params_c, params_g, state_g, opt_g_state, c_grads_dp, c_batch_ids, prng_key):
        (gen_loss, stage_g), g_grad = jax.value_and_grad(loss_fn_g, has_aux=True)(params_g, state_g, params_c, c_grads_dp, c_batch_ids, prng_key)

        updates_g, opt_g_state = opt_g.update(g_grad, opt_g_state)
        params_g = optax.apply_updates(params_g, updates_g)

        return params_g, state_g, opt_g_state, gen_loss

    return update_g


def latest_params(config):
    files = glob.glob(os.path.join(config.results_dir,"training_grads/", "g_*.pickle"))
    files.sort(key= lambda x: int(os.path.basename(x).strip('g_').strip(".pickle")))
    grad_path = files[-1]
    with open(grad_path, 'rb') as f:
        g_struct = pickle.load(f)
    return g_struct

'''
Uses the grads and params from the classifier during training to train
a generator.
'''
def train_generator(c: configlib.Config, generaator_prng_seed: int = 1):
    if c.remove_extreme_grads:
        grad_dataset = dataset.GradSubset(os.path.join(c.results_dir, "training_grads"))
    else:
        grad_dataset = dataset.GradDataset(os.path.join(c.results_dir, "training_grads"))

    forward_classifier = get_classifier(c)
    forward_generator = get_generator(c, n_batches=len(grad_dataset))

    # Get config parameters from json dict
    beta_1_g = c["beta1_g"]
    beta_2_g = c["beta2_g"]
    batch_size_c = c.batch_size

    model_c = hk.without_apply_rng(hk.transform(forward_classifier))

    # Get initial params and state for generator
    g_prng_seq = hk.PRNGSequence(generaator_prng_seed)
    model_g = hk.transform_with_state(forward_generator)

    params_g, state_g = model_g.init(next(g_prng_seq), batch_size_c, is_training=True)

    # Initialize optimizer for generator
    opt_g = optax.adam(learning_rate=c.g_lr, b1=beta_1_g, b2=beta_2_g)
    opt_g_state = opt_g.init(params_g)

    # Get initial params for generator
    (images, labels), _ = model_g.apply(params_g, state_g, next(g_prng_seq), batch_size_c, False)
    _ = model_g.apply(params_g, state_g, next(g_prng_seq), batch_size_c, False)

    grad_dataloader = GradLoader(
        grad_dataset,
        batch_size=c.g_batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Getting the test loader for evaluation
    _, _, _, test_loader = get_dataset(c)

    # Train generator with saved gradients
    _train_generator(c,
            grad_dataloader,
            model_g,
            opt_g,
            params_g,
            state_g,
            opt_g_state,
            model_c,
            g_prng_seq,
            test_loader=test_loader)

def _train_generator(config: configlib.Config, loader, model_g, opt_g, params_g, state_g, opt_g_state, model_c, prng_seq, test_loader=None):
    z_dim = config.z_dim
    clip = config.dp_clip
    im_chan = config["im_chan"]
    b_size_g_gen = config.g_generated_batch_size if config.g_generated_batch_size is not None else config.batch_size
    b_size_g_train = config.g_batch_size
    n_classes = config.n_classes


    loss_fn_c = get_classifier_training_loss_fn(config, model_c)

    loss_fn_g = get_g_loss(config,
        model_g, z_dim, n_classes, im_chan, loss_fn_c, clip, b_size_g_train, b_size_g_gen,
        min_loss_n=config["min_loss_n"]
    )
    update_g = get_update_g(model_c, model_g, loss_fn_g, opt_g)

    gen_dir = get_gen_dir(config)
    os.makedirs(gen_dir, exist_ok=True)

    # Functions for synthetic accuracy evaluation
    loss_fn = get_classifier_training_loss_fn(config, model_c)
    synthetic_accuracy = get_synthetic_accuracy(config=config,
                                                test_loader=test_loader,
                                                model_g=model_g)
    syn_accuracies = []
    for e in range(config.g_epochs):
        tot_loss = step = 0
        pbar_it = tqdm(loader, disable=not config.progress_bar)
        for data in pbar_it:
            step += 1
            c_grad_dp = data["grads"]
            c_params_dp = data["params"]
            c_batch_ids = data["batch_id"]

            # Compute classifier grads on generated data and update generator

            if config.batch_conditioning:
                # Duplicate the batch_ids to match the desired batch size and layout for
                # generation.
                c_batch_ids = jnp.repeat(c_batch_ids, b_size_g_gen)
                c_batch_ids = jnp.tile(c_batch_ids, config["min_loss_n"])

            params_g, state_g, opt_g_state, g_loss = update_g(c_params_dp, params_g, state_g, opt_g_state, c_grad_dp, c_batch_ids, next(prng_seq))
            tot_loss += g_loss
            pbar_it.set_description(f"Epoch={e} Loss={g_loss:.4f} Avg. loss={tot_loss/step:.4f}")

        # Save generator after each epochs
        if config.offline_save_all_epoch_generators:
            gen_path = os.path.join(gen_dir, f'trained_gen_epoch_{e}.pickle')
            with open(gen_path, 'wb') as f:
                g_dict = {"params": params_g, "state":state_g}
                pickle.dump(g_dict, f)

        if test_loader is not None:
            syn_accuracy = synthetic_accuracy(params_g=params_g, state_g=state_g, prng_key=next(prng_seq))
            syn_accuracies.append(syn_accuracy)
            print(f"Generator Epoch : {e}, Generator accuracy : {syn_accuracy}")


        generate_and_save_images(config, model_g, params_g, state_g, next(prng_seq), prefix=str(e))

    save_training_info(save_dir = config.results_dir, syn_accuracy=syn_accuracies)
    print("\nFinished training offline generator.\n")


def save_training_info(save_dir, syn_accuracy):
    """save training information"""
    eval_info_path = os.path.join(save_dir, "syn_eval_info.pickle")
    print(f'saving {eval_info_path}')

    with open(eval_info_path, 'wb') as f:
        eval_dict = {"synthetic_accuracy" : syn_accuracy,
                    }
        pickle.dump(eval_dict, f)


def get_gen_dir(config):
    """Returns directory where the generator parameters are saved"""
    return os.path.join(config.results_dir, 'generators/')

