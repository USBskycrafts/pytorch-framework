import torch
import torch.nn as nn
from torch.optim import lr_scheduler


class Playground:
    def __init__(self,
                 config,
                 models,
                 optimizers,
                 writer
                 ):
        self.config = config
        self.models = models
        self.optimizers = optimizers

        step_size = config.getint("train", "step_size")
        gamma = config.getfloat("train", "lr_multiplier")
        self.schedulers = list(map(lambda optimizer: lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma), self.optimizers))
        self.writer = writer
        self.train_step = 0
        self.test_step = 0
        self.eval_step = 0

    def train(self, data, config, gpu_list, acc_result, mode):
        raise NotImplementedError("Please implement this method")

    def _train(self, *args, **kwargs):
        map(lambda model: model.train(), self.models.values())
        map(lambda optimizer: optimizer.zero_grad(), self.optimizers)
        result = self.train(*args, **kwargs)
        self.train_step += 1
        losses, acc_result = result["losses"], result["acc_result"]
        map(lambda loss: loss.backward(), losses)
        map(lambda optimizer: optimizer.step(), self.optimizers)
        map(lambda scheduler: scheduler.step(), self.schedulers)
        return {
            "loss": sum([loss.float() for loss in losses]),
            "acc_result": acc_result
        }

    def test(self, data, config, gpu_list, acc_result, mode):
        raise NotImplementedError("Please implement this method")

    def _test(self, *args, **kwargs):
        map(lambda model: model.eval(), self.models.values())
        result = self.test(*args, **kwargs)["output"]
        self.test_step += 1
        return result

    def eval(self, data, config, gpu_list, acc_result, mode):
        raise NotImplementedError("Please implement this method")

    def _eval(self, *args, **kargs):
        map(lambda model: model.eval(), self.models.values())
        result = self.eval(*args, **kargs)
        self.eval_step += 1
        losses, acc_result = result["losses"], result["acc_result"]
        return {
            "loss": sum([loss.float() for loss in losses]),
            "acc_result": acc_result
        }
