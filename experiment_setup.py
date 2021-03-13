import argparse
import torch
from torch.utils.data import Dataset
import model
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
                        action="store_false",
                        default=True,
                        dest="mu_init_eval",
                        help='Init mu multiple times and use best init.')
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

class custom_dataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, labels=None):
        self.data_tensor = data_tensor
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.labels = labels
        if labels is not None:
            assert data_tensor.shape[0] == labels.shape[0]

    def __getitem__(self, index):
        if self.labels is None:
            return self.data_tensor[index].to(self.device)
        else:
            return self.data_tensor[index].to(self.device), self.labels[index].to(self.device)

    def __len__(self):
        return self.data_tensor.size(0)


def load_model(model_dir, x_train, device="cuda"):
    """ load model with appropriate settings """
    checkpoint = join_path(model_dir, 'best.pth.tar')
    if not os.path.exists(checkpoint):
        raise Exception("File {} dosen't exists!".format(checkpoint))
    ckpt = torch.load(checkpoint, map_location=device)
    saved_dict = ckpt['state_dict']
    beta = ckpt.get("beta_override")
    sigma = ckpt.get("sigma")
    net = model.VAE_bodies(x_train, ckpt['layers'], num_components=ckpt['num_components'], device=device)
    net.init_std(x_train, gmm_mu=torch.Tensor(ckpt['gmm_means']), gmm_cv=torch.Tensor(ckpt['gmm_cv']),
                 weights=ckpt['weights'], inv_maxstd=ckpt['inv_maxstd'], beta_constant=ckpt['beta_constant'],
                 beta_override=beta, sigma=sigma)
    saved_dict = ckpt['state_dict']
    new_dict = net.state_dict()
    new_dict.update(saved_dict)
    net.load_state_dict(new_dict)
    net.eval()
    net.to(device)
    return net


def load_checkpoint(checkpoint, model, cpu=False):
    if not os.path.exists(checkpoint):
        raise Exception("File {} dosen't exists!".format(checkpoint))
    if cpu:
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint)
    return checkpoint


def save_checkpoint(state, save_dir, ckpt_name='best.pth.tar'):
    file_path = os.path.join(save_dir, ckpt_name)
    if not os.path.exists(save_dir):
        print("Save directory dosen't exist! Making directory {}".format(save_dir))
        os.mkdir(save_dir)
    torch.save(state, file_path)

def make_dir(directory_path):
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
        print_stdout("Directory {} created.".format(directory_path))

class LatentVariablesDataset(Dataset):
    @property
    def num_factors(self):
        raise NotImplementedError()

    @staticmethod
    def ground_truth_to_classes(ground_truth_factors):
        raise NotImplementedError()

    @property
    def factors_keys(self):
        raise NotImplementedError()

    def sample(self, batch_size):
        raise NotImplementedError()


