import json
import jax.random
import optax
import haiku as hk
import utils
import training.training_config
import training.offline.classifier
import training.offline.generator
import training.online.classifier
import os
from classifiers.classifiers import get_classifier
from generators.generators import get_generator
import sys
from data_utils.datasets import get_dataset
import configlib

# Configuration arguments
parser = configlib.add_parser("Run config")
parser.add_argument("--gpu_id", type=int,
        help="Force the use of a specfici GPU id.")
parser.add_argument("--progress_bar", default=False, action='store_true',
        help="Enable progress bars (e.g. when training locally).")

parser.add_argument("--train_classifier", default=False, action='store_true',
        help="Train the classifier.")
parser.add_argument("--train_generator", default=False, action='store_true',
        help="Train the generator.")

parser.add_argument("--train_online", default=False, action='store_true',
        help="Train online classifier.")

parser.add_argument("--results_dir", type=str, default='./results',
        help="The directory where to store results and data produced by the run.")

#parser.add_argument("--g_tv_l1", type=float, default=None)
#parser.add_argument("--g_tv_l2", type=float, default=None)
#parser.add_argument("--g_l2_reg", type=float, default=None)
parser.add_argument("--g_fid_reg", type=float, default=None)

def main():
    c = configlib.parse(save_fname="last_arguments.txt")
    configlib.print_config(c)

    # To create a config ``c'' from a dict ``conf'' (e.g. for parameter tuning):
    #     c = Config(**conf)
    # Or:
    #     c = Config()
    #     c.update(conf)

    utils.ensure_dir(c.results_dir)
    with open(os.path.join(c.results_dir, "config.json"), 'w') as f:
        f.write(json.dumps(c))

    if c.gpu_id: os.environ["CUDA_VISIBLE_DEVICES"] = c.gpu_id

    if c.train_classifier:
        training.offline.classifier.train_classifier(c)

    if c.train_generator:
        training.offline.generator.train_generator(c)

    if c.train_online:
        training.online.classifier.train_classifier(c)

if __name__ == "__main__":
    main()

