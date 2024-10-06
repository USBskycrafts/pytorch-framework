import logging
import torch
import os
import json
from playground.playground import Playground
from reader.reader import init_dataset, init_formatter, init_test_dataset
from model import get_model
from model.optimizer import init_optimizer
from playground import get_playground
from .output_init import init_output_function
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def init_all(config, gpu_list, checkpoint, mode, *args, **params):

    result = {}
    logger.info("Begin to initialize dataset and formatter...")
    if mode == "train":
        # init_formatter(config, ["train", "valid"], *args, **params)
        result["train_dataset"], result["valid_dataset"] = init_dataset(
            config, *args, **params)
    else:
        # init_formatter(config, ["test"], *args, **params)
        result["test_dataset"] = init_test_dataset(config, *args, **params)

    logger.info("Begin to initialize models...")
    model_name = config.get("model", "model_name").replace(" ", "").split(",")
    models = {name: get_model(name)(
        config, gpu_list, *args, **params) for name in model_name}
    optimizer_names = config.get(
        "model", "optimizer_name").replace(" ", "").split(",")
    optimizers = {model_name: init_optimizer(model, config, optimizer_name,
                                             *args, **params)
                  for (model_name, model), optimizer_name in zip(models.items(), optimizer_names)}
    trained_epoch = 0
    global_step = 0

    if len(gpu_list) > 0:
        models = {name: model.cuda()
                  for name, model in models.items()}

        try:
            # map(lambda model: model.init_multi_gpu(
            #     gpu_list, config, *args, **params), models)
            [model.init_multi_gpu(
                gpu_list, config, *args, **params) for model in models.values()]

        except Exception as e:
            logger.warning(
                "No init_multi_gpu implemented in the model, use single gpu instead.")

    try:
        parameters = torch.load(checkpoint)
        for model_name, model in models.items():
            model.load_state_dict(parameters[model_name])
        # map(lambda model_name, model: model.load_state_dict(
        #     parameters[model_name]), models.items())

        if mode == "train":
            trained_epoch = parameters["trained_epoch"]
            for model_name, optimizer in optimizers.items():
                optimizer.load_state_dict(
                    parameters[optimizer.__class__.__name__ + model_name])
            if "global_step" in parameters:
                global_step = parameters["global_step"]
    except Exception as e:
        information = "Cannot load checkpoint file with error %s" % str(e)
        if mode == "test":
            logger.error(information)
            raise e
        else:
            logger.warning(information)

    writer = SummaryWriter(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")),
                           config.get("output", "model_name"))
    playground_name = config.get("model", "playground")
    playground = get_playground(playground_name)(
        config, models, optimizers, writer)
    result["playground"] = playground
    if mode == "train":
        result["trained_epoch"] = trained_epoch
        result["output_function"] = init_output_function(config)
        result["global_step"] = global_step

    logger.info("Initialize done.")

    return result
