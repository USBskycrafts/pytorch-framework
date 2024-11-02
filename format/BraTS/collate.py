import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from format.dictionary.dictionary_collate import DictionaryCollateFormatter
from format.augment.batch_process import batch_process


class BraTSCollateFormatter(DictionaryCollateFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError(
            'BraTS atlas not matched due to torchvision randomly transform limitation. A bug exits. Please use `BraTSBaseFormatter`')

    def process(self, data, config, mode, *args, **kwargs):
        batch = super().process(data, config, mode, *args, **kwargs)
        if mode == 'train':
            atlas = {k: v for k, v in batch.items() if k in [
                'T1', 'T2', 'T1ce']}
            atlas = batch_process(atlas, transforms.Compose([
                transforms.Resize(
                    (256, 256), interpolation=F.InterpolationMode.BILINEAR),
                transforms.RandomApply(
                    [transforms.RandomAffine(
                        degrees=(-180, 180),
                        scale=(0.8, 1.2),
                        interpolation=F.InterpolationMode.BILINEAR)],
                    p=0.2
                ),
                transforms.RandomHorizontalFlip(p=0.1),
                transforms.RandomVerticalFlip(p=0.1),
            ]))
            batch['T1'], batch['T2'], batch['T1ce'] = atlas['T1'], atlas['T2'], atlas['T1ce']
        else:
            atlas = {k: v for k, v in batch.items() if k in [
                'T1', 'T2', 'T1ce']}
            atlas = batch_process(atlas, transforms.Compose([
                transforms.Resize(
                    (256, 256), interpolation=F.InterpolationMode.BILINEAR),
            ]))
            batch['T1'], batch['T2'], batch['T1ce'] = atlas['T1'], atlas['T2'], atlas['T1ce']
        return batch


class BraTSBaseFormatter(DictionaryCollateFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, data, config, mode, *args, **kwargs):
        batch = super().process(data, config, mode, *args, **kwargs)
        atlas = {k: v for k, v in batch.items() if k in [
            'T1', 'T2', 'T1ce']}
        atlas = batch_process(atlas, transforms.Compose([
            transforms.Resize(
                (256, 256), interpolation=F.InterpolationMode.BILINEAR),
        ]))
        batch['T1'], batch['T2'], batch['T1ce'] = atlas['T1'], atlas['T2'], atlas['T1ce']
        return batch
