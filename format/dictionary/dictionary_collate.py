import torch
from format.Basic import BasicFormatter


class DictionaryCollateFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **kwargs):
        super().__init__(config, mode, *args, **kwargs)

    def process(self, data, config, mode, *args, **kwargs):
        batch = {}
        for sample in data:
            for key, value in sample.items():
                if key not in batch:
                    batch[key] = []
                batch[key].append(value)
        for k in batch:
            batch[k] = torch.stack(batch[k], dim=0)
        return batch
