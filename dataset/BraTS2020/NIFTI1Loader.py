import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import logging
import nibabel as nib
from typing import List


class NIFTI1Loader(Dataset):
    def __init__(self, config, mode, *args, **kwargs):
        super().__init__()
        self.config = config
        self.mode = mode
        self.t1_dir = config.get("data", "%s_t1_dir" % mode)
        self.t2_dir = config.get("data", "%s_t2_dir" % mode)
        self.t1ce_dir = config.get("data", "%s_t1ce_dir" % mode)
        self.logger = logging.getLogger(__name__)
        self.input_dim = config.getint("model", "input_dim")
        self.output_dim = config.getint("model", "output_dim")
        self.data_list = []

        for (T1, T2, T1CE) in zip(sorted(os.listdir(self.t1_dir)),
                                  sorted(os.listdir(self.t2_dir)),
                                  sorted(os.listdir(self.t1ce_dir))):
            if T1.endswith(".nii") and T2.endswith(".nii") and T1CE.endswith(".nii"):
                def load_from_path(dir, path):
                    path = os.path.join(dir, path)
                    image = nib.nifti1.load(path)
                    # transform to standard pytorch tensor
                    tensor = torch.Tensor(image.get_fdata())
                    tensor = tensor.permute(2, 0, 1)
                    # normalize
                    tensor = (tensor - tensor.min()) / \
                        (tensor.max() - tensor.min())
                    return tensor
                T1, T2, T1CE = map(load_from_path, [
                    self.t1_dir, self.t2_dir, self.t1ce_dir], [T1, T2, T1CE])
                T1, T2, T1CE = map(lambda x: self.data_process(
                    x, config, mode, *args, **kwargs), [T1, T2, T1CE])

                self.data_list.extend({
                    "t1": t1,
                    "t2": t2,
                    "t1ce": t1ce,
                } for (t1, t2, t1ce) in zip(T1, T2, T1CE))

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)

    def data_process(self, data: torch.Tensor, config, mode, *args, **params) -> List[torch.Tensor]:
        # TODO: 1. crop from 155 to 16 in channels(using the middle half channels)
        #       2. resize the shape from 240x240 to 64x64 for training
        data = self.size_process(data, config, mode, *args, **params)
        data_list = self.channel_process(data, config, mode, *args, **params)
        return data_list

    def channel_process(self, data: torch.Tensor, config, mode, *args, **params) -> List[torch.Tensor]:
        n_channels = data.shape[0]
        start = n_channels // 2 - n_channels // 4
        end = n_channels // 2 + n_channels // 4
        data = data[start:end, :, :]

        input_dim = config.getint("model", "input_dim")
        data_list = list(data.split(input_dim, dim=0))
        if data_list[-1].shape[0] < input_dim:
            data_list.pop()
        return data_list

    def size_process(self, data: torch.Tensor, config, mode, *args, **params) -> torch.Tensor:
        if mode == 'train':
            return data[:, 50:200, 30:220]
        elif mode == 'valid':
            return data[:, 120-32:120+32, 120-32:120+32]
        else:
            return data
