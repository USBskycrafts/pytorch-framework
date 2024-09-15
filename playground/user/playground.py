from torch.nn import BCELoss, L1Loss
from playground.playground import Playground
import torch
from model.user.net import UserNet
from tools.accuracy_tool import general_image_metrics
from model.loss import GramLoss


class UserPlayground(Playground):
    def __init__(self, config,
                 models,
                 optimizers,
                 writer):
        super().__init__(config, models, optimizers, writer)

        self.generator = models['UserNet']
        self.discriminator = models['discriminator']
        self.l1_loss = L1Loss()
        self.gram_loss = GramLoss()

    def train(self, data, config, gpu_list, acc_result, mode):
        pred = self.generator(torch.cat([data['t1'], data['t2']], dim=1))
        target = data['t1ce']
        fake_features = self.discriminator(pred)
        g_loss = self.l1_loss(
            pred, target) + self.l1_loss(fake_features[-1], torch.ones_like(fake_features[-1]))
        acc_result = general_image_metrics(pred, target, config, acc_result)
        yield {
            "name": "UserNet",
            "loss": g_loss,
            "acc_result": acc_result
        }
        fake_features = self.discriminator(pred.detach())
        target_features = self.discriminator(target)
        d_loss = 0.5 * (self.l1_loss(fake_features[-1],
                                     torch.zeros_like(fake_features[-1]))
                        + self.l1_loss(target_features[-1],
                                       torch.ones_like(target_features[-1])))
        d_loss += self.gram_loss(fake_features, target_features)
        yield {
            "name": "discriminator",
            "loss": d_loss,
            "acc_result": acc_result
        }
        self.writer.add_scalar("UserNet", g_loss, self.train_step)
        self.writer.add_scalar("discriminator", d_loss, self.train_step)

    def test(self, data, config, gpu_list, acc_result, mode):
        pred = self.generator(torch.cat([data['t1'], data['t2']], dim=1))
        target = data['t1ce']
        acc_result = general_image_metrics(pred, target, config, acc_result)
        return {
            "output": [acc_result],
            "acc_result": acc_result
        }

    def eval(self, data, config, gpu_list, acc_result, mode):
        pred = self.generator(torch.cat([data['t1'], data['t2']], dim=1))
        target = data['t1ce']
        loss = self.l1_loss(pred, target)
        acc_result = general_image_metrics(pred, target, config, acc_result)
        return {
            "loss": loss,
            "acc_result": acc_result
        }
