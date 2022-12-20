import os
from tqdm import tqdm
import pickle
import jax.random
import haiku as hk
import jax.numpy as jnp
import optax
import utils
import dp_accountant
from data_utils.datasets import get_dataset
from data_utils import dataset
from data_utils.dataloader import GradLoader
from classifiers.classifiers import get_classifier
from generators.generators import get_generator
from training.optimizers import get_classifier_optimizer
from training.utils import get_c_loss_real, get_accuracy, clip_grads, grads_norm_fn, grad_norm_fn, noise_grads, update_model, per_input_grads, sum_grads, rescale_grads, get_compute_grads, combine_grads, generate_and_save_images

from training.loss_functions import get_classifier_training_loss_fn, get_gan_classifier_loss, get_gan_generator_loss

import configlib


def setup_generator(c: configlib.Config, prng_seq):
    forward_generator = get_generator(c)
    model_g = hk.transform_with_state(forward_generator)
    params_g, state_g = model_g.init(next(prng_seq), c.batch_size, is_training=True)

    return model_g, params_g, state_g

def save_training_info(save_dir, eps, classifier_accuracy):
    """save training information"""
    eval_info_path = os.path.join(save_dir, "classifier_eval_info.pickle")
    print(f'saving {eval_info_path}')

    with open(eval_info_path, 'wb') as f:
        eval_dict = {"epsilon": eps, "classifier_accuracy" : classifier_accuracy,
                    }
        pickle.dump(eval_dict, f)


'''
Trains a classifier and saves the grads and params of the classifier
during training for each step.
'''
def train_classifier(c, model_prng_seed=0, gan_generator_prng_seed=1):
    forward_classifier = get_classifier(c)

    batch_size_c = c.batch_size

    train_dataset, test_dataset, train_loader, test_loader = get_dataset(c)
    dataset_size = len(train_dataset)

    # Get initial params for classifier
    c_prng_seq = hk.PRNGSequence(model_prng_seed)
    model_c = hk.without_apply_rng(hk.transform(forward_classifier))

    img, _ = next(iter(train_loader))
    img_size = img.shape[-3:]
    params_c = model_c.init(rng=next(c_prng_seq), x=img)

    # Initialize optimizer for classifier
    epoch_length = len(train_loader)
    opt_c = get_classifier_optimizer(c, epoch_length)

    opt_c_state = opt_c.init(params_c)

    # Train classifier and save gradients
    q = batch_size_c / dataset_size

    early_stop = False
    if c.c_max_steps is not None and c.c_max_steps > 0:
        early_stop = True
    noise_multiplier = c.dp_noise_multiplier
    clip = c.dp_clip
    sigma = noise_multiplier * clip
    delta = c.dp_delta
    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    loss_fn = get_classifier_training_loss_fn(c, model_c)
    if c.gan:
        gen_loss_fn = get_gan_classifier_loss(c, model_c, 0)

    if c.gan:
        # Setup the generator and all required functions for it
        g_prng_seq = hk.PRNGSequence(gan_generator_prng_seed)

        model_g, params_g, state_g = setup_generator(c, g_prng_seq)
        opt_g = optax.adam(learning_rate=c.gan_lr)
        opt_g_state = opt_g.init(params_g)

        gan_gen_loss_fn = get_gan_generator_loss(c, model_g, model_c)
        clipped_grads_c = get_compute_grads(
                c, gen_loss_fn, do_clip_grads=not c.disable_dp, do_noise_grads=False, get_metadata=False)
        gan_gen_grads = jax.value_and_grad(gan_gen_loss_fn, has_aux=True)

    c_accuracies = []
    epsilons = []

    accuracy = get_accuracy(c, model_c)
    dp_grads_c = get_compute_grads(c, loss_fn, do_clip_grads=not c.disable_dp, do_noise_grads=not c.disable_dp)

    continue_training = True
    steps = 0
    for e in range(c.c_epochs):
        if not continue_training:
            break

        step = 0
        tot_loss = 0
        pbar_it = tqdm(train_loader, disable=not c.progress_bar)
        for real_img, real_label in pbar_it:
            if early_stop and steps >= c.c_max_steps:
                continue_training = False
                break

            step += 1

            real_label = jnp.expand_dims(real_label, axis=1)
            c_grad_dp, loss, grads_norm, clipped_grads_norm, avg_grads_norm = dp_grads_c(
                params_c, real_img, real_label, sigma, clip, next(c_prng_seq)
            )

            grads_bellow_size = None
            if c.c_min_grad_size is not None: grads_bellow_size = jnp.sum(grads_norm < c.c_min_grad_size)
            grads_bellow_size = None
            if c.c_max_grad_size is not None: grads_above_size = jnp.sum(grads_norm > c.c_max_grad_size)

            utils.save_grads_and_params(
                c_grad_dp, params_c, steps, os.path.join(c.results_dir, "training_grads"),
                grads_bellow_size=grads_bellow_size, grads_above_size=grads_above_size
            )

            if c.gan:
                fake_batch_size = c.batch_size if c.gan_discr_batch_size is None else c.gan_discr_batch_size

                (fake, labels), _ = model_g.apply(
                        params_g, state_g, next(g_prng_seq), fake_batch_size, is_training=False
                )
                labels = jnp.expand_dims(labels, axis=1)
                c_grad_fake = clipped_grads_c(params_c, fake, labels, clip=clip)
                grads = combine_grads(c_grad_dp, c_grad_fake, c.batch_size, fake_batch_size)
            else:
                grads = c_grad_dp

            params_c, opt_c_state = update_model(opt_c, opt_c_state, grads, params_c)
            tot_loss += loss
            steps += 1

            if c.gan:
                for _ in range(c.gan_gen_steps):
                    (g_loss, state_g), g_grads = gan_gen_grads(params_g, state_g, params_c, next(g_prng_seq))
                    params_g, opt_g_state = update_model(opt_g, opt_g_state, g_grads, params_g)

            # Calculate Rényi-DP budget used
            eps_loss, alpha = dp_accountant.calculate_privacy_loss(q, steps, alphas, sigma, delta, clip)
            desc = f"E={e} Loss={loss:.2f} Avg. loss={tot_loss/step:.4f} ε={eps_loss:.2f} α={alpha} ||∇||=[{jnp.sum(grads_norm < 1)} {jnp.min(grads_norm):.2f}-{jnp.percentile(grads_norm,0.95):.2f}-{jnp.max(grads_norm):.2f}] AvgNorm={avg_grads_norm:.2f}"
            if c.gan:
                desc += f" LossG={g_loss:.2f} ||∇G||={grad_norm_fn(g_grads):.2f}"
            pbar_it.set_description(desc)

        if test_loader is not None:
            correct = 0
            tot = 0
            pbar_it = tqdm(test_loader, disable=not c.progress_bar)
            for real_img, real_label in pbar_it:
                _correct, _tot = accuracy(params_c, real_img, jnp.expand_dims(real_label, axis=1))
                correct += _correct
                tot += _tot
                pbar_it.set_description(f"Epoch={e} Acc={100*correct/tot:.2f} Eps={eps_loss:.2f} Alpha={alpha}")
            c_accuracies.append(correct/tot)
            #print(f"Classifier step : {step}, Classifier accuracy : {correct/tot}")
            #print(f"correct : {correct}, tot : {tot}")
        epsilons.append(eps_loss)

        if c.gan:
            generate_and_save_images(c, model_g, params_g, state_g, next(g_prng_seq), prefix=f"gan_e{e}")

    print(f'saving in {c.results_dir}')
    save_training_info(save_dir = c.results_dir,
                        eps = epsilons,
                        classifier_accuracy=c_accuracies)
    print("\nFinished training offline classifier.\n")


