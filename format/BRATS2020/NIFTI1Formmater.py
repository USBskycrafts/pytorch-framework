import logging
from typing import Dict, List
import torch
from format.Basic import BasicFormatter


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
        number_list = [number for number in map(lambda x: x['number'], data)]
        layer_list = [layer for layer in map(lambda x: x['layer'], data)]
        # print(number_list, layer_list)
        return {
            't1': torch.stack(t1_list, dim=0),
            't2': torch.stack(t2_list, dim=0),
            't1ce': torch.stack(t1ce_list, dim=0),
            'number': torch.stack(number_list, dim=0),
            'layer': torch.stack(layer_list, dim=0)
        }
