import torch

class MuGridDataset(torch.utils.data.Dataset):
    def __init__(self, endpoint_tensor, device="cuda"):
        self.endpoint_tensor = endpoint_tensor
        self.device = device

    def __getitem__(self, index):
            return self.endpoint_tensor[index], index

    def __len__(self):
        return self.endpoint_tensor.size(0)