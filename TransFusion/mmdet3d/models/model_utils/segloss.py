import torch
import torch.nn as nn
import cv2
import numpy as np
from ..losses.auxseg_loss import SEGLOSS
import torch.nn.functional as F




class Gaussian(nn.Module):
    def __init__(self, **kwarg):
        super(Gaussian, self).__init__()
        self.num_classes = kwarg['num_classes']
        self.out_channel = kwarg['out_channels']
        self.aux_img_cls = nn.Conv2d(self.out_channel, self.num_classes+1, 1)
        self.init_weights()
        self.AUX_LOSS = SEGLOSS(**kwarg['seg_loss_cfg'])

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.aux_img_cls.bias, -np.log((1 - pi) / pi))

    def pred_fg_img(self, img_feat):
        pred_fg_img = self.aux_img_cls(img_feat.unsqueeze(dim=0))
        return pred_fg_img

    def forward(self, img_feat, batch_dict, cam_key, _idx):
        fg_pred = self.pred_fg_img(img_feat)
        batch_dict['fg_pred'] = fg_pred
        eps = 1e-8
        input_soft = F.softmax(fg_pred, dim=1) + eps
        if "gt_boxes2d" in batch_dict:
            self.auxloss = self.AUX_LOSS(batch_dict, cam_key, _idx)
        else:
            self.auxloss = None
        return self.auxloss, input_soft


__all__ = {
    "Gaussian": Gaussian,
}
