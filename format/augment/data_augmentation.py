import torch
import numpy as np
from typing import Dict
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.abstract_transforms import RndTransform


def get_data_augmentation(tensor):
    bs, c, h, w = tensor.shape
    batch = DataLoader(tensor, bs)
    spatial_transform = SpatialTransform((h, w), np.array((h, w)) // 2,
                                         do_elastic_deform=True, alpha=(0., 1500.), sigma=(30., 50.),
                                         do_rotation=True, angle_z=(0, 2 * np.pi),
                                         do_scale=True, scale=(0.5, 2),
                                         border_mode_data='constant', border_cval_data=0, order_data=1,
                                         random_crop=True)
    spatial_transform = RndTransform(spatial_transform, prob=0.5)
    multithreaded_generator = SingleThreadedAugmenter(
        batch, Compose([spatial_transform]))
    return multithreaded_generator


class DataLoader(SlimDataLoaderBase):
    def __init__(self, data, BATCH_SIZE=2, num_batches=None, seed=False):
        super(DataLoader, self).__init__(data, BATCH_SIZE, num_batches)
        # data is now stored in self._data.

    def generate_train_batch(self):
        # usually you would now select random instances of your data. We only have one therefore we skip this
        img = self._data

        # now construct the dictionary and return it. np.float32 cast because most networks take float
        return {'data': img}
