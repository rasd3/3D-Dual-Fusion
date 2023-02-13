
import torch
import torch.nn as nn
import cv2
import numpy as np
from pcdet.utils import common_utils


class devil(nn.Module):
    def __init__(self):
        super(devil, self).__init__()
        self.scale = 1

        self.img_channels = [256,512,1024]
        self.pts_channels = [32,64,64]
        self.ld = 0.5
        self.beta = 0.1
        self.img_list = [nn.ModuleList(),nn.ModuleList()]
        self.pts_list = nn.ModuleList()
        self.conv = nn.ModuleList()
        for idx in range(self.scale):
            for sp in range(2):
                self.img_list[sp].append(
                    nn.Conv2d(self.img_channels[idx],
                            self.img_channels[idx],
                            kernel_size=3,
                            stride=1,
                            padding=1)
                )
            self.conv.append(
                nn.Conv2d(self.img_channels[idx],
                        self.img_channels[idx],
                        kernel_size=3,
                        stride=1,
                        padding=1)
            )
            self.pts_list.append(
                nn.Conv2d(self.pts_channels[idx],
                          self.img_channels[idx],
                          kernel_size=3,
                          stride=1,
                          padding=1)
            )

    def forward(self,
        img_feats,
        pts_feats
        ):
        '''
        s_r : self-reflect
        m_r : mutual_reflect
        '''
        device = img_feats[0].device
        batch_size = img_feats[0].shape[0]
        rgb_list = []
        for idx in range(self.scale):
            scale_img_feats = img_feats[idx]
            scale_pts_feats = pts_feats[idx]

            img_channel = scale_img_feats.shape[1]
            pts_channel = scale_pts_feats.shape[1]

            s_r_scale_img_feats = self.img_list[0][idx].to(device=device)(scale_img_feats)
            m_r_scale_img_feats = self.img_list[1][idx].to(device=device)(scale_img_feats)
            s_r_scale_pts_feats = self.pts_list[idx].to(device=device)(scale_pts_feats)

            s_r_scale_img_feats_flatten = s_r_scale_img_feats.reshape(batch_size, img_channel, -1)
            m_r_scale_img_feats_flatten = m_r_scale_img_feats.reshape(batch_size, img_channel, -1)
            s_r_scale_pts_feats_flatten = s_r_scale_pts_feats.reshape(batch_size, img_channel, -1)

            s_r_reflect_output = torch.bmm(torch.transpose(s_r_scale_img_feats_flatten,1,2).contiguous(),m_r_scale_img_feats_flatten)
            m_r_reflect_output = torch.bmm(torch.transpose(s_r_scale_pts_feats_flatten,1,2).contiguous(),m_r_scale_img_feats_flatten)

            s_r_reflect_output = torch.sigmoid(s_r_reflect_output)
            m_r_reflect_output = torch.sigmoid(m_r_reflect_output)

            reflect_output = self.ld * s_r_reflect_output + (1-self.ld) *m_r_reflect_output

            conv_img_feats = self.conv[idx].to(device=device)(scale_img_feats)
            conv_img_feats_flatten = conv_img_feats.reshape(batch_size, img_channel, -1)

            conv_img_output = torch.bmm(conv_img_feats_flatten,torch.transpose(reflect_output,1,2).contiguous())

            output = scale_img_feats + self.beta * conv_img_output

            rgb_list.append(output)
        return rgb_list


class BasicGate(nn.Module):
    def __init__(self, g_channel_list, sparse_shape, voxel_size, point_cloud_range, inv_idx, num_conv = 2):
        super(BasicGate, self).__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.inv_idx = inv_idx
        self.g_channel_list = g_channel_list
        self.sparse_shape = sparse_shape
        self.spatial_basic_list = []
        for scale in range(len(self.g_channel_list)):
            self.spatial_basic = []
            for idx in range(num_conv - 1):
                self.spatial_basic.append(
                    nn.Conv2d(self.g_channel_list[scale],
                            self.g_channel_list[scale],
                            kernel_size=3,
                            stride=1,
                            padding=1))
                self.spatial_basic.append(
                    nn.BatchNorm2d(self.g_channel_list[scale], eps=1e-3, momentum=0.01)
                )
                self.spatial_basic.append(
                    nn.ReLU()
                )
            self.spatial_basic.append(
                nn.Conv2d(self.g_channel_list[scale], 1, kernel_size=3, stride=1, padding=1))
            self.spatial_basic_list.append(nn.Sequential(*self.spatial_basic))


        self.sigmoid = nn.Sigmoid()
    def forward(self, x_rgb, x_list, batch_dict):
        batch_size = batch_dict['batch_size']
        calibs = batch_dict['calib']
        scale_size = len(x_rgb)
        device = x_rgb[0].device
        img_shapes = [
            torch.tensor(f.shape[2:], device=device) for f in x_rgb
        ]
        h, w = batch_dict['images'].shape[2:]
        
        pts_img_list = []
        for s in range(scale_size):
            batch_img_list = []
            batch_index = x_list[s].indices[:, 0]
            ratio = self.sparse_shape[1] / x_list[s].spatial_shape[1]
            spatial_indices = x_list[s].indices[:, 1:] * ratio
            voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3]
            
            for b in range(batch_size):
                calib = calibs[b]
                voxels_3d_batch = voxels_3d[batch_index==b]
                voxel_features_sparse = x_list[s].features[batch_index==b]

                # Reverse the point cloud transformations to the original coords.
                if 'noise_scale' in batch_dict:
                    voxels_3d_batch[:, :3] /= batch_dict['noise_scale'][b]
                if 'noise_rot' in batch_dict:
                    voxels_3d_batch = common_utils.rotate_points_along_z(voxels_3d_batch[:, self.inv_idx].unsqueeze(0), -batch_dict['noise_rot'][b].unsqueeze(0))[0, :, self.inv_idx]
                if 'flip_x' in batch_dict:
                    voxels_3d_batch[:, 1] *= -1 if batch_dict['flip_x'][b] else 1
                if 'flip_y' in batch_dict:
                    voxels_3d_batch[:, 2] *= -1 if batch_dict['flip_y'][b] else 1

                voxels_2d, _ = calib.lidar_to_img(voxels_3d_batch[:, self.inv_idx].cpu().numpy())
                voxels_2d_norm = voxels_2d / np.array([w, h])
                voxels_2d_norm_tensor = torch.Tensor(voxels_2d_norm).to(device)
                voxels_2d_norm_tensor = torch.clamp(voxels_2d_norm_tensor, min=0.0, max=1.0)
                pt_img, _ = pts2img(voxels_2d_norm_tensor, voxel_features_sparse, img_shapes[s], voxels_3d_batch)
                batch_img_list.append(pt_img.unsqueeze(0))
            pts_img_list.append(torch.cat(batch_img_list))
        new_img_feats = []
        
        for idx in range(len(pts_img_list)):
            pts = self.spatial_basic_list[idx].to(device=device)(pts_img_list[idx])
            attention_map = torch.sigmoid(pts)
            new_img_feats.append(x_rgb[idx] * attention_map)

        return new_img_feats

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

attn_dict = {
    'BiGate1D': BiGate1D,
    'BiGate1D_2': BiGate1D_2,
    'BiGateSum1D': BiGateSum1D,
    'BiGateSum1D_2': BiGateSum1D_2,
}
