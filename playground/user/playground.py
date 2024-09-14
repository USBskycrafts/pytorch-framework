from torch.nn import BCELoss
from playground.playground import Playground
import torch
from model.user.net import UserNet


class UserPlayground(Playground):
    def __init__(self, config,
                 models,
                 optimizers,
                 writer):
        super().__init__(config, models, optimizers, writer)

        self.generator = models['UserNet']
        self.discriminator = models['discriminator']
        self.bce_loss = BCELoss()

    def train(self, data, config, gpu_list, acc_result, mode):
        g_result = self.generator(data, config, gpu_list, acc_result, mode)
        predict, g_loss = g_result['predict'], g_result['loss']
        d_result = self.discriminator(
            {
                "fake": predict.detach(),
                "real": data['t1ce']
            },
            config, gpu_list, acc_result, mode)

        pred_fake = d_result['pred']['fake']
        g_loss += self.bce_loss(pred_fake, torch.ones_like(pred_fake))
        d_loss = d_result['loss']

        self.writer.add_scalar('loss/g_loss', g_loss,
                               global_step=self.train_step)
        self.writer.add_scalar('loss/d_loss', d_loss,
                               global_step=self.train_step)

        return {
            "losses": [g_loss, d_loss],
            "acc_result": g_result['acc_result']
        }

    def test(self, data, config, gpu_list, acc_result, mode):
        g_result = self.generator(data, config, gpu_list, acc_result, mode)
        return g_result

    def eval(self, data, config, gpu_list, acc_result, mode):
        g_result = self.generator(data, config, gpu_list, acc_result, mode)
        self.writer.add_scalar('loss/g_loss', g_result['loss'],
                               global_step=self.eval_step)
        return {
            **g_result,
            "losses": [g_result['loss']],
        }
