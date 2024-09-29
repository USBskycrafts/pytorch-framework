from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW
from adam_mini import Adam_mini


def init_optimizer(model, config, optimizer_type, *args, **params):
    learning_rate = config.getfloat("train", "learning_rate")
    if optimizer_type == "adam":
        optimizer = Adam(model.parameters(), lr=learning_rate,
                         weight_decay=config.getfloat("train", "weight_decay"))
    elif optimizer_type == "sgd":
        optimizer = SGD(model.parameters(), lr=learning_rate,
                        weight_decay=config.getfloat("train", "weight_decay"))
    elif optimizer_type == "adam_mini":
        optimizer = Adam_mini(
            model.named_parameters(),
            lr=learning_rate,
            weight_decay=config.getfloat("train", "weight_decay"),
            dim=config.getint("model", "dim"),
            n_heads=config.getint("model", "n_heads"),
        )
    elif optimizer_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=learning_rate,
                          weight_decay=config.getfloat("train", "weight_decay"))
    else:
        raise NotImplementedError

    return optimizer
