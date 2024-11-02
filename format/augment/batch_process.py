import torchvision.transforms as transforms


def batch_process(batch, transform):
    for k in batch:
        batch[k] = transform(batch[k])
    return batch
