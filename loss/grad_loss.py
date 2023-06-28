from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

#
# def measure_pixelwise_gradient(pred, conf_thresh_lower=None, conf_thresh_upper=None):
#     """This version uses numpy for creating empty tensors dtype=float64
#     This is performing the best keep this for now
#     """
#     count = 0
#     batch_gradient = np.zeros((pred.shape[0], 8, 224, 224))
#     pred_sigmoid = torch.sigmoid(pred)
#
#     for zz in range(0, pred.shape[0]):
#         pred_clip = pred_sigmoid[zz][0]
#         if conf_thresh_lower is not None:
#             pred_clip[pred_clip < conf_thresh_lower] = 0
#         if conf_thresh_upper is not None:
#             pred_clip[pred_clip > conf_thresh_upper] = 1
#
#         clip_gradient = np.gradient(np.gradient(pred_clip.cpu().detach().numpy(), axis=0), axis=0)
#         clip_gradient -= clip_gradient.min()
#         clip_gradient /= (clip_gradient.max() - clip_gradient.min() + 1e-7)
#
#         batch_gradient[zz] = clip_gradient
#
#     batch_gradient = torch.from_numpy(batch_gradient)
#
#     return batch_gradient
#
# class Gradient_Loss(nn.Module):
#
#     def __init__(self):
#         self.l1 = nn.L1Loss(reduction='none')
#
#     def forward(self, sr, hr):
#         batch_grad = measure_pixelwise_gradient(sr)
#         batch_grad = batch_grad.type(torch.cuda.FloatTensor)
#
#         loss = torch.mean(batch_grad*self.l1(sr, hr))
#         return loss


class CentralDifferent(nn.Module):
    def __init__(self):
        super(CentralDifferent, self).__init__()
        in_ch = 3
        k_s = 3
        self.group = 3
        self.l1 = nn.L1Loss(reduction='none')
        self.weight = torch.ones((in_ch, 1, k_s, k_s)).cuda()
        self.weights_c = self.weight.sum(dim=[2, 3], keepdim=True)

    def forward(self, sr, hr, mask):
        def CDL(x):
            yc = F.conv2d(x, self.weights_c, stride=1, padding=0, groups=self.group)
            y = F.conv2d(x, self.weight, stride=1, padding=1, dilation=1, groups=self.group)
            return y - yc

        mask = mask.detach()
        mask = mask.sum(dim=1, keepdim=True)
        mask_mean = mask.mean(dim=[2,3], keepdim=True)
        mask = (mask - mask_mean >= 0.0).float()
        sr_c = CDL(sr)
        hr_c = CDL(hr)

        return torch.mean(self.l1(sr_c, hr_c) * mask.detach())