
import torch
import torch.nn as nn
import cv2
import numpy as np

class BiGate(nn.Module):
    def __init__(self, g_channel, g_channel_):
        super(BiGate, self).__init__()
        self.g_channel = g_channel
        self.g_channel_ = g_channel_
        self.b_conv2d = nn.Conv2d(self.g_channel,
                                  1,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        self.a_conv2d = nn.Conv2d(self.g_channel_,
                                  1,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
    def forward(self, feat1, feat2):
        feat1_map = self.b_conv2d(feat1)
        feat1_scale = torch.sigmoid(feat1_map)
        feat2_map = self.a_conv2d(feat2)
        feat2_scale = torch.sigmoid(feat2_map)
        return feat1 * feat2_scale, feat2 * feat1_scale

class BiGate1D(nn.Module):
    def __init__(self, g_channel, g_channel_):
        super(BiGate1D, self).__init__()
        self.g_channel = g_channel
        self.g_channel_ = g_channel_
        self.b_conv1d = nn.Conv1d(self.g_channel,
                                  1,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.a_conv1d = nn.Conv1d(self.g_channel_,
                                  1,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
    def forward(self, feat1, feat2):
        feat1_map = self.b_conv1d(feat1.permute(0, 2, 1)).permute(0, 2, 1)
        feat1_scale = torch.sigmoid(feat1_map)
        feat2_map = self.a_conv1d(feat2.permute(0, 2, 1)).permute(0, 2, 1)
        feat2_scale = torch.sigmoid(feat2_map)
        return feat1 * feat2_scale, feat2 * feat1_scale

class BiGate1D_2(nn.Module):
    def __init__(self, g_channel, g_channel_):
        super(BiGate1D_2, self).__init__()
        self.g_channel = g_channel
        self.g_channel_ = g_channel_
        self.b_conv1d = nn.Conv1d(self.g_channel,
                                  1,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.a_conv1d = nn.Conv1d(self.g_channel_,
                                  1,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
    def forward(self, feat1, feat2):
        fuse_feat = feat1.permute(0, 2, 1) + feat2.permute(0, 2, 1)
        feat1_map = self.b_conv1d(fuse_feat).permute(0, 2, 1)
        feat1_scale = torch.sigmoid(feat1_map)
        feat2_map = self.a_conv1d(fuse_feat).permute(0, 2, 1)
        feat2_scale = torch.sigmoid(feat2_map)
        return feat1 * feat1_scale, feat2 * feat2_scale

class BiGateSum1D(nn.Module):
    def __init__(self, g_channel, g_channel_):
        super(BiGateSum1D, self).__init__()
        self.g_channel = g_channel
        self.g_channel_ = g_channel_
        self.b_conv1d = nn.Conv1d(self.g_channel,
                                  1,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.a_conv1d = nn.Conv1d(self.g_channel_,
                                  1,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
    def forward(self, feat1, feat2):
        feat1_map = self.b_conv1d(feat1.permute(0, 2, 1)).permute(0, 2, 1)
        feat1_scale = torch.sigmoid(feat1_map)
        feat2_map = self.a_conv1d(feat2.permute(0, 2, 1)).permute(0, 2, 1)
        feat2_scale = torch.sigmoid(feat2_map)
        return feat1 + (feat2 * feat1_scale), feat2 + (feat1 * feat2_scale)

class BiGateSum1D_2(nn.Module):
    def __init__(self, g_channel, g_channel_):
        super(BiGateSum1D_2, self).__init__()
        self.g_channel = g_channel
        self.g_channel_ = g_channel_
        self.b_conv1d = nn.Conv1d(self.g_channel,
                                  1,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.a_conv1d = nn.Conv1d(self.g_channel_,
                                  1,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
    def forward(self, feat1, feat2):
        fuse_feat = feat1.permute(0, 2, 1) + feat2.permute(0, 2, 1)
        feat1_map = self.b_conv1d(fuse_feat).permute(0, 2, 1)
        feat1_scale = torch.sigmoid(feat1_map)
        feat2_map = self.a_conv1d(fuse_feat).permute(0, 2, 1)
        feat2_scale = torch.sigmoid(feat2_map)
        return feat1 + (feat2 * feat1_scale), feat2 + (feat1 * feat2_scale)

def pts2img(coor, pts_feat, shape, pts, ret_depth=False):
    def visualize(pts_feat):
        pts_feat = pts_feat.detach().cpu().max(2)[0].numpy()
        pts_feat = (pts_feat * 255.).astype(np.uint8)
        cv2.imwrite('lidar2img.png', pts_feat)

    coor = coor[:, [1, 0]]
    i_shape = torch.cat(
        [shape + 1,
            torch.tensor([pts_feat.shape[1]]).cuda()])
    i_pts_feat = torch.zeros(tuple(i_shape), device=coor.device)
    i_coor = (coor * shape).to(torch.long)
    i_pts_feat[i_coor[:, 0], i_coor[:, 1]] = pts_feat
    i_pts_feat = i_pts_feat[:-1, :-1].permute(2, 0, 1)
    #visualize(i_pts_feat)
    if ret_depth:
        i_shape[2] = 3
        i_depth_feat = torch.zeros(tuple(i_shape), device=coor.device)
        i_depth_feat[i_coor[:, 0], i_coor[:, 1]] = pts
        i_depth_feat = i_depth_feat[:-1, :-1].permute(2, 0, 1)
        return i_pts_feat, i_depth_feat
    
    return i_pts_feat, None


attn_dict = {
    'BiGate1D': BiGate1D,
    'BiGate1D_2': BiGate1D_2,
    'BiGateSum1D': BiGateSum1D,
    'BiGateSum1D_2': BiGateSum1D_2,
}
