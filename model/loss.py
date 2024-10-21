import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor
from torchvision.ops import sigmoid_focal_loss
import numpy as np


class MultiLabelSoftmaxLoss(nn.Module):
    def __init__(self, config):
        super(MultiLabelSoftmaxLoss, self).__init__()
        self.task_num = config.getint("model", "output_dim")
        self.criterion = []
        for a in range(0, self.task_num):
            try:
                ratio = config.getfloat("train", "loss_weight_%d" % a)
                self.criterion.append(
                    nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, ratio], dtype=np.float32)).cuda()))
                # print_info("Task %d with weight %.3lf" % (task, ratio))
            except Exception as e:
                self.criterion.append(nn.CrossEntropyLoss())

    def forward(self, outputs, labels):
        loss = 0
        for a in range(0, len(outputs[0])):
            o = outputs[:, a, :].view(outputs.size()[0], -1)
            loss += self.criterion[a](o, labels[:, a])

        return loss


def multi_label_cross_entropy_loss(outputs, labels):
    labels = labels.float()
    temp = outputs
    res = - labels * torch.log(temp) - (1 - labels) * torch.log(1 - temp)
    res = torch.mean(torch.sum(res, dim=1))

    return res


def cross_entropy_loss(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    return criterion(outputs, labels)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class BCELoss2d(nn.Module):
    """
    Binary Cross Entropy loss function
    """

    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1)
        return self.bce_loss(logits_flat, labels_flat)


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


class DiceLoss(nn.Module):
    def __init__(self, multiclass=False):
        super(DiceLoss, self).__init__()
        self.multiclass = multiclass

    def forward(self, input, target):
        return dice_loss(input, target, multiclass=self.multiclass)


class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, pred, mask):
        return self.wbce_loss(pred, mask)

    def wbce_loss(self, pred, mask):
        weight = 1 + 5 * \
            torch.abs(F.avg_pool2d(mask, kernel_size=31,
                                   stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce=None)
        wbce = (weight*wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
        return wbce.mean()


class FocalLoss2d(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        return sigmoid_focal_loss(input, target, self.alpha, self.gamma, self.reduction)


class SobelLoss(nn.Module):
    def __init__(self):
        super(SobelLoss, self).__init__()
        self.l1_criterion = nn.L1Loss()

    def sobel_x(self, x):
        sobel_kernel = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x, sobel_kernel, padding=1)

    def sobel_y(self, x):
        sobel_kernel = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x, sobel_kernel, padding=1)

    def forward(self, pred, target):
        return self.l1_criterion(self.sobel_x(pred), self.sobel_x(target)) + self.l1_criterion(self.sobel_y(pred), self.sobel_y(target))
