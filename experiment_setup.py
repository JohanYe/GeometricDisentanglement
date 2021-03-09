import argparse
import ast
import datetime
import os
from os.path import join as join_path




def parse_args(default_params: dict) -> dict:
    """
    Parse arguments from command line.

    Args:
        default_params (dict): the dictionary containing the default parameter values. It aso implicitely defines
                                which parameters can be parsed.
    Returns:
        parsed_params (dict): the dictionary containing the parameters for the experiments after having been parsed.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model_dir',
                        type=str,
                        metavar='MODEL_DIR',
                        dest="model_dir",
                        default=default_params["model_dir"],
                        help="path to model to fit land to")
    parser.add_argument('--dataset',
                        type=str,
                        metavar='DATASET',
                        dest="dataset",
                        default=default_params["dataset"],
                        help="identifier of the dataset to be used, available dataset: bodies, MNIST.")
    parser.add_argument('--exp_name',
                        type=str,
                        metavar='EXP_NAME',
                        dest="exp_name",
                        default=default_params["exp_name"],
                        help="Experiment name inside model folder")
    parser.add_argument("--sampled",
                        action="store_false",
                        default=True,
                        dest="sampled",
                        help='Usage of sampling for constnt estimation, strongly recommend.')
    parser.add_argument("--load_land",
                        action="store_true",
                        default=False,
                        dest="load_land",
                        help='Loading previous training, used for resuming training of model.')
    parser.add_argument("--hpc",
                        action="store_true",
                        default=False,
                        dest="hpc",
                        help='Used to reduce batch size for local testing.')
    parser.add_argument("--mu_init_eval",
                        action="store_true",
                        default=False,
                        dest="mu_init_eval",
                        help='Init mu multiple times and use best init otherwise we use Sturms mean.')
    parser.add_argument("--debug_mode",
                        action="store_true",
                        default=False,
                        dest="debug_mode",
                        help='Used to only train on subset of data to facilitate easy debugging.')
    args = parser.parse_args()
    print(args)
    parsed_params = default_params.copy()
    for name, value in vars(args).items():
        parsed_value = value
        if name in list(default_params.keys()):
            parsed_params[name] = parsed_value
        else:
            raise Exception(
                "Parameter {} with value {} was not recognized, available parameters: {}".format(name, value,
                                                                                                 list(
                                                                                                     default_params.keys())))
    return parsed_params
