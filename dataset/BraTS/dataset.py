import torch
import os
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any
from tqdm import tqdm


class BraTSDataset(Dataset):
    def __init__(self, config, mode, *args, **kwargs):
        super().__init__()
        self.data_dir = config.get("data", "%s_data_dir" % mode)
        self.data = []
        for root, dirs, files in os.walk(self.data_dir):
            if not any([file.endswith('.nii') for file in files]):
                continue
            sample_name = os.path.basename(root)
            num = int(sample_name.split('_')[-1])
            record: Dict[str, Any] = {
                'number': num
            }
            for file in files:
                if 't1ce' in file:
                    atlas = nib.nifti1.load(
                        os.path.join(root, file)).get_fdata()
                    # normalize the data
                    atlas = self.normalize(atlas)
                    record['T1ce'] = atlas
                elif 't1' in file:
                    atlas = nib.nifti1.load(
                        os.path.join(root, file)).get_fdata()
                    # normalize the data
                    atlas = self.normalize(atlas)
                    record['T1'] = atlas
                elif 't2' in file:
                    atlas = nib.nifti1.load(
                        os.path.join(root, file)).get_fdata(dtype=np.float32)
                    # normalize the data
                    atlas = self.normalize(atlas)
                    record['T2'] = atlas
                else:
                    continue
            for layer, modals in enumerate(zip(
                    torch.split(record['T1'], 1, dim=0),
                    torch.split(record['T1ce'], 1, dim=0),
                    torch.split(record['T2'], 1, dim=0))):
                self.data.append({
                    'number': torch.tensor(record['number']),
                    'layer': torch.tensor(layer),
                    'T1': modals[0],
                    'T1ce': modals[1],
                    'T2': modals[2],
                })
        print("Total %d samples" % len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def normalize(self, atlas):
        atlas -= atlas.min()
        atlas = (atlas / np.percentile(atlas, 99)).clip(0, 1)

        # transform to torch.Tensor
        atlas = torch.from_numpy(atlas).float().permute(2, 0, 1)
        return atlas
