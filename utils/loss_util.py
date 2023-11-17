import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import joblib
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
import math
from abc import abstractmethod

import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from tensorboardX import SummaryWriter


Tensor = torch.Tensor

class Criterion(object):
    def __init__(self, config:dict, writer: SummaryWriter):
        self.config = config
        self.criterion = config['loss']
        self.device = config['device'] if torch.cuda.is_available() else 'cpu'
        self.writer = writer

    @abstractmethod
    def __call__(self, input: Tensor, target: Tensor, gt_target=None, clean_mask=None,
                 reduction: str='mean', idx=None, test=False, epoch=None, step=None) -> Tensor:
        pass

    def flush(self, epoch, postfix=''):
        return None

class SimpleLoss(Criterion):
    """
    loss init and inference
    """
    def __init__(self, config: dict, writer: SummaryWriter):
        super(SimpleLoss, self).__init__(config, writer)
        if self.criterion == 'bmse':
            self.init_bmse()

        elif self.criterion == 'weighted_mse':
            self.normalize_batch = self.config['loss_params']['weighted_mse']['normalize_batch']

    def __call__(self, input: Tensor, target: Tensor, gt_target=None, clean_mask=None,
                 reduction: str='mean', idx=None, test=False, epoch=None, step=None) -> Tensor:
        if self.criterion in ['mse', 'l2']:
            target = target.view(-1, 1)
            return F.mse_loss(input, target, reduction=reduction)

        elif self.criterion in ['l1']:
            target = target.view(-1, 1)
            return F.l1_loss(input, target, reduction=reduction)

        elif self.criterion in ['ce', 'crossentropy']:
            target = target.to(torch.int64)
            return F.cross_entropy(input, target, reduction=reduction)

        elif self.criterion in ['sce']:
            return self.sce_loss(input, target, reduction, test=test)

        else:
            raise NotImplementedError(f"NotImplemented loss type {self.criterion}")


    def sce_loss(self, input: Tensor, target: Tensor, reduction: str, test=False):
        """
        (Wang, ICCV 2019) Symmetric Cross Entropy for Robust Learning with Noisy Labels
        https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels
        l_sl = alpha*l_ce + beta*l_rce
        A = log(eps_rce) -> -4 following paper
        """
        self.alpha = self.config['loss_params']['sce']['alpha']
        self.beta = self.config['loss_params']['sce']['beta']
        y_hat_1 = F.softmax(input, dim=1)
        y_1 =  F.one_hot(target, num_classes=input.shape[1])

        y_hat_2 = y_hat_1
        y_2 = y_1

#        y_hat_1 = torch.clamp(y_hat_1, min=1e-7, max=1.0)
        y_2 = torch.clamp(y_2, min=1e-4, max=1.0)

        loss = -self.alpha * torch.sum(y_1 * torch.log(y_hat_1), dim=1) \
            -self.beta * torch.sum(y_2 * torch.log(y_hat_2), dim=1)

        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'none':
            return loss
        else:
            raise NotImplementedError(f"NotImplemented reduction type: {reduction}")


class SceLoss(Criterion):
    """
    (Wang, ICCV 2019) Symmetric Cross Entropy for Robust Learning with Noisy Labels
    https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels
    l_sl = alpha*l_ce + beta*l_rce
    A = log(eps_rce) -> -4 following paper
    """
    def __call__(self, input: Tensor, target: Tensor, gt_target=None, clean_mask=None,
                 reduction: str='mean', idx=None, test=False, epoch=None, step=None) -> Tensor:
        self.alpha = self.config['loss_params']['sce']['alpha']
        self.beta = self.config['loss_params']['sce']['beta']
        y_hat = F.softmax(input, dim=1)
        y =  F.one_hot(target, num_classes=input.shape[1])

        y_hat_clamped = torch.clamp(y_hat, min=1e-7, max=1.0)
        y_clamped = torch.clamp(y, min=1e-4, max=1.0) # min=1e-4 from A=-4

        loss_ce = torch.sum(y * torch.log(y_hat_clamped), dim=1)
        loss_rce = torch.sum(y_hat * torch.log(y_clamped), dim=1)
        loss = - self.alpha * loss_ce - self.beta * loss_rce

        if step != None:
            self.writer.add_scalar('sce/train/loss_ce', torch.mean(loss_ce), step)
            self.writer.add_scalar('sce/train/loss_rce', torch.mean(loss_rce), step)

        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'none':
            return loss
        else:
            raise NotImplementedError(f"NotImplemented reduction type: {reduction}")


CRITERION = {
    'mse': SimpleLoss,
    'l2': SimpleLoss,
    'l1': SimpleLoss,
    'ce': SimpleLoss,
    'sce': SceLoss,
}
