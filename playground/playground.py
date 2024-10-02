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
        self.schedulers = {model_name: lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma) for model_name, optimizer in self.optimizers.items()}
        self.writer = writer
        self.train_step = 0
        self.test_step = 0
        self.eval_step = 0

    def train(self, data, config, gpu_list, acc_result, mode):
        raise NotImplementedError("Please implement this method")

    def _train(self, *args, **kwargs):
        # map(lambda model: model.train(), self.models.values())
        [model.train() for model in self.models.values()]
        trainer = self.train(*args, **kwargs)
        self.train_step += 1
        loss = []
        for result in trainer:
            self.optimizers[result["name"]].zero_grad()
            _loss, acc_result = result["loss"], result["acc_result"]
            loss.append(_loss)
            _loss.backward()
            self.optimizers[result["name"]].step()
            self.schedulers[result["name"]].step()
        return {
            "loss": sum(loss),
            "acc_result": acc_result
        }

    def test(self, data, config, gpu_list, acc_result, mode):
        raise NotImplementedError("Please implement this method")

    def _test(self, *args, **kwargs):
        [model.eval() for model in self.models.values()]
        result = self.test(*args, **kwargs)
        self.test_step += 1
        return result

    def eval(self, data, config, gpu_list, acc_result, mode):
        raise NotImplementedError("Please implement this method")

    def _eval(self, *args, **kargs):
        # map(lambda model: model.eval(), self.models.values())
        [model.eval() for model in self.models.values()]
        result = self.eval(*args, **kargs)
        self.eval_step += 1
        loss, acc_result = result["loss"], result["acc_result"]
        return {
            "loss": loss,
            "acc_result": acc_result
        }
