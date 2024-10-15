import torch.optim as optim
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW


def init_optimizer(model, config, *args, **params):
    optimizer_type = config.get("train", "optimizer")
    learning_rate = config.getfloat("train", "learning_rate")
    if optimizer_type == "adam":
        optimizer = Adam(model.parameters(), lr=learning_rate,
                         weight_decay=config.getfloat("train", "weight_decay"))
    elif optimizer_type == "sgd":
        optimizer = SGD(model.parameters(), lr=learning_rate,
                        weight_decay=config.getfloat("train", "weight_decay"),
                        momentum=config.getfloat("train", "momentum"),
                        nesterov=config.getboolean("train", "nesterov"))

    elif optimizer_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=learning_rate,
                          weight_decay=config.getfloat("train", "weight_decay"))
    else:
        raise NotImplementedError

    return optimizer
