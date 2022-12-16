import jax
import jax.numpy as jnp
import haiku as hk

import configlib
from classifiers.classifiers import get_classifier
from training.utils import get_accuracy, get_c_loss_real, get_compute_grads, update_model
from training.optimizers import get_classifier_optimizer

parser = configlib.add_parser("Evaluation config")
parser.add_argument("--eval_classifier_model", default="cnn_small", type=str, metavar="EVAL_MODEL_NAME",
        help="The type of classifier to train (cnn_small, cnn_med) for synthetic data evaluation.")
parser.add_argument("--eval_c_lr", type=float, default=0.1, metavar="EVAL_LR",
        help="The classifier's (starting) learning rate for synthetic data evaluation.")
parser.add_argument("--eval_c_activation", default="relu", type=str, metavar="ACTIVATION_NAME",
        help="The type of classifier to train (relu, leaky_relu, tanh, elu) for synthetic data evaluation.")
parser.add_argument("--eval_negative_slope", default=.2, type=float,
        help="The negative slope when using leaky_relu for synthetic data evaluation.")
parser.add_argument("--eval_elu_alpha", default=1., type=float,
        help="The alpha param of ELU activations (when ELU is used) for synthetic data evaluation.")
parser.add_argument("--eval_classifier_optimizer", default="sgd", type=str,
        help="The optimizer to train the classifier (sgd, ) for synthetic data evaluation.")
parser.add_argument("--eval_c_momentum", type=float, default=0.0, metavar="EVAL_GAMMA",
        help="The classifier's training momentum for synthetic data evaluation (when applicable).")
parser.add_argument("--eval_c_epochs", type=int, default=1,
        help="Number of ecpochs to train the classifier for synthetic data evaluation.")
parser.add_argument("--eval_syn_size", default=128, type=int,
        help="Size of the generated synthetic dataset for classifier for synthetic data evaluation.")


def get_synthetic_accuracy(config, test_loader, model_g):
    """Returns a function that returns the accuracy of a classifier trained on generated data"""
    # Create a config to call the classifier functions
    eval_classifier_params = {'classifier_model' : config.eval_classifier_model,
                                'n_classes': config.n_classes,
                                'c_lr': config.eval_c_lr,
                                'c_activation': config.eval_c_activation,
                                'negative_slope': config.eval_negative_slope,
                                'elu_alpha': config.eval_elu_alpha,
                                'classifier_optimizer': config.eval_classifier_optimizer,
                                'c_momentum': config.eval_c_momentum,
                                'c_epochs' : config.eval_c_epochs,
                                'disable_dp' : True}
    eval_classifier_config = configlib.Config(eval_classifier_params)
    #eval_classifier_config = config
    print("Eval config")
    configlib.print_config(eval_classifier_config)
    forward_classifier = get_classifier(eval_classifier_config)
    model_c = hk.without_apply_rng(hk.transform(forward_classifier))
    accuracy = get_accuracy(eval_classifier_config, model_c)
    loss_fn = get_c_loss_real(model_c)
    syn_grads_c = get_compute_grads(eval_classifier_config, loss_fn, do_clip_grads=False, do_noise_grads=False)
    opt_c = get_classifier_optimizer(eval_classifier_config, config.eval_batch_size)
    n_epochs = config.eval_c_epochs
    batch_size = config.eval_syn_size

    img, _ = next(iter(test_loader))

    @jax.jit
    def synthetic_accuracy(params_g, state_g, prng_key):
        params_c = model_c.init(rng=prng_key, x=img)
        opt_c_state = opt_c.init(params_c)
        for _ in range(n_epochs):
            (syn_img, syn_label), _ = model_g.apply(params_g, state_g, prng_key, batch_size=batch_size, is_training=False, batches=None)
            syn_label = jnp.expand_dims(syn_label, axis=1)
            syn_grad, _, _, _, _ = syn_grads_c(
                params = params_c,
                inputs=syn_img,
                labels=syn_label,
                sigma=None,
                clip=None,
                prng_key=prng_key
            )
            params_c, opt_c_state = update_model(opt_c, opt_c_state, syn_grad, params_c)
            
        correct = 0
        tot = 0
        for real_img, real_label in test_loader:
            _correct, _tot = accuracy(params_c, real_img, jnp.expand_dims(real_label, axis=1))
            correct += _correct
            tot += _tot
        return correct/tot
    return synthetic_accuracy