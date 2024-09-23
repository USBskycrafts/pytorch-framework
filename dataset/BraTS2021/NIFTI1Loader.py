import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import logging
import nibabel as nib
from typing import List


class NIFTI1Loader(Dataset):
    def __init__(self, config, mode, *args, **kwargs):
        training_num = config.getint("dataset", "training_num")
        eval_num = config.getint("dataset", "eval_num")
        test_num = config.getint("dataset", "test_num")
        dataset_path = config.get("dataset", "BraTS2021")
        self.norm = config.getint("data", "normalization")
        data_list = os.listdir(dataset_path)
        assert len(data_list) == training_num + eval_num + \
            test_num, "dataset size is not correct: " + str(len(data_list))
        if mode == "train":
            data_list = data_list[:training_num]
        elif mode == "valid":
            data_list = data_list[training_num:training_num+eval_num]
        elif mode == "test":
            data_list = data_list[training_num +
                                  eval_num:training_num+eval_num+test_num]
        else:
            logging.error("mode must be train, valid or test")
            raise ValueError("mode must be train, valid or test")
        dataset = []
        for item in data_list:
            item = os.path.join(dataset_path, item)
            if os.path.isdir(item):
                modals = {}
                for file in os.listdir(item):
                    if "t1ce" in file:
                        file_path = os.path.join(dataset_path, item, file)
                        data = nib.nifti1.load(file_path)
                        modals["t1ce"] = self.transform_tensor(
                            data.get_fdata(), mode)
                    elif "t1" in file:
                        file_path = os.path.join(dataset_path, item, file)
                        data = nib.nifti1.load(file_path)
                        modals["t1"] = self.transform_tensor(
                            data.get_fdata(), mode)
                    elif "t2" in file:
                        file_path = os.path.join(dataset_path, item, file)
                        data = nib.nifti1.load(file_path)
                        modals["t2"] = self.transform_tensor(
                            data.get_fdata(), mode)

                # now modals is a dictionary of lists
                dataset.extend([dict(zip(modals.keys(), values))
                               for values in zip(*modals.values())])
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def transform_tensor(self, data, mode):
        tensor = torch.Tensor(data)
        tensor = (tensor - tensor.min()) * self.norm / \
            (tensor.max() - tensor.min())
        h, w, c = tensor.shape
        tensor = tensor[:, :, c // 4: c // 4 * 3]
        if mode != "test":
            tensor = tensor[40:200, 30:220, :]
        tensor = tensor.permute(2, 0, 1)
        return torch.split(tensor, 1, dim=0)
