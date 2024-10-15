import logging
from typing import Dict, List
import torch
from format.Basic import BasicFormatter
from format.augment.data_augmentation import get_data_augmentation


class NIFTI1Formatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.config = config
        self.mode = mode
        self.logger = logging.getLogger(__name__)

    def process(self, data: List[Dict[str, torch.Tensor]], config, mode, *args, **params):
        t1_list = [t1 for t1 in map(lambda x: x['t1'], data)]
        t2_list = [t2 for t2 in map(lambda x: x['t2'], data)]
        t1ce_list = [t1ce for t1ce in map(lambda x: x['t1ce'], data)]
        mask_list = [mask for mask in map(lambda x: x['mask'], data)]
        number_list = [number for number in map(lambda x: x['number'], data)]
        layer_list = [number for number in map(lambda x: x['layer'], data)]
        batch = {
            't1': torch.stack(t1_list, dim=0),
            't2': torch.stack(t2_list, dim=0),
            't1ce': torch.stack(t1ce_list, dim=0),
            'mask': torch.stack(mask_list, dim=0),
            'number': torch.stack(number_list, dim=0),
            'layer': torch.stack(layer_list, dim=0),
        }
        if mode != 'test':
            aug = get_data_augmentation(
                torch.cat([batch['t1'], batch['t2'],
                          batch['t1ce'], batch['mask']], dim=1).numpy(),
            )
            aug = next(aug)
            aug['data'] = torch.from_numpy(aug['data'])
            t1, t2, t1ce, mask = torch.split(aug['data'], 1, dim=1)
            assert batch['t1'].shape == t1.shape
            batch['t1'] = t1
            batch['t2'] = t2
            batch['t1ce'] = t1ce
            batch['mask'] = mask
        return batch
