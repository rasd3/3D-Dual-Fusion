
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
    def __init__(self, img_channel_list, pts_channel_list, sparse_shape, voxel_size, point_cloud_range, inv_idx, pts_idx, num_conv = 2):
        super(BasicGate, self).__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.inv_idx = inv_idx
        self.g_channel_list = pts_channel_list
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
                def visualize(rgb_feature,batch_dict,b):
                    cv2.imwrite('image.jpg',batch_dict['images'][b].permute(1,2,0).cpu().numpy()*255)
                    rgb_feature = rgb_feature.cpu().detach().numpy()
                    min = rgb_feature.min()
                    max = rgb_feature.max()
                    rgb_feature = (rgb_feature-min)/(max-min)
                    rgb_feature = rgb_feature*255
                    max_pts_feat = np.max(np.transpose(rgb_feature.astype("uint8"),(1,2,0)),axis=2)
                    max_pts_feat = cv2.applyColorMap(max_pts_feat,cv2.COLORMAP_JET)
                    cv2.imwrite("feature.jpg",max_pts_feat)
                voxels_2d, _ = calib.lidar_to_img(voxels_3d_batch[:, self.inv_idx].cpu().numpy())
                voxels_2d_norm = voxels_2d / np.array([w, h])
                voxels_2d_norm_tensor = torch.Tensor(voxels_2d_norm).to(device)
                voxels_2d_norm_tensor = torch.clamp(voxels_2d_norm_tensor, min=0.0, max=1.0)
                #visualize(x_rgb[0][b],batch_dict,b)
                pt_img, _ = pts2img(voxels_2d_norm_tensor, voxel_features_sparse, img_shapes[s], voxels_3d_batch)
                batch_img_list.append(pt_img.unsqueeze(0))
            pts_img_list.append(torch.cat(batch_img_list))
        new_img_feats = []
        
        for idx in range(len(pts_img_list)):
            pts = self.spatial_basic_list[idx].to(device=device)(pts_img_list[idx])
            attention_map = torch.sigmoid(pts)
            new_img_feats.append(x_rgb[idx] * attention_map)

        return new_img_feats

class BasicGatev2(nn.Module):
    def __init__(self, img_channel_list, pts_channel_list, sparse_shape, voxel_size, point_cloud_range, inv_idx, pts_idx, num_conv = 2):
        super(BasicGatev2, self).__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.inv_idx = inv_idx
        self.img_channel_list = img_channel_list
        self.pts_channel_list = pts_channel_list
        self.sparse_shape = sparse_shape
        self.spatial_basic_list = []
        self.channel_reduce_list = []
        for scale in range(len(self.pts_channel_list)):
            self.spatial_basic = []
            for idx in range(num_conv - 1):
                self.spatial_basic.append(
                    nn.Conv2d(self.pts_channel_list[scale],
                            self.pts_channel_list[scale],
                            kernel_size=3,
                            stride=1,
                            padding=1))
                self.spatial_basic.append(
                    nn.BatchNorm2d(self.pts_channel_list[scale], eps=1e-3, momentum=0.01)
                )
                self.spatial_basic.append(
                    nn.ReLU()
                )
            self.spatial_basic.append(
                nn.Conv2d(self.pts_channel_list[scale], 1, kernel_size=3, stride=1, padding=1))
            self.spatial_basic_list.append(nn.Sequential(*self.spatial_basic))
            self.channel_reduce_list.append(nn.Conv2d(self.pts_channel_list[scale], self.img_channel_list[scale], kernel_size=1, stride=1))
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
            new_img_feats.append(x_rgb[idx] + self.channel_reduce_list[idx].to(device=device)(attention_map * pts_img_list[idx]))

        return new_img_feats

class BasicGatev3(BasicGatev2):
    def __init__(self, img_channel_list, pts_channel_list, sparse_shape, voxel_size, point_cloud_range, inv_idx, pts_idx, num_conv = 2):
        super(BasicGatev3, self).__init__(img_channel_list, pts_channel_list, sparse_shape, voxel_size, point_cloud_range, inv_idx, num_conv)
        
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
            new_img_feats.append(attention_map * x_rgb[idx] + self.channel_reduce_list[idx].to(device=device)(attention_map * pts_img_list[idx]))

        return new_img_feats

class Patch(BasicGatev2):
    def __init__(self, img_channel_list, pts_channel_list, sparse_shape, voxel_size, point_cloud_range, inv_idx, pts_idx, num_conv = 2):
        super(Patch, self).__init__(img_channel_list, pts_channel_list, sparse_shape, voxel_size, point_cloud_range, inv_idx, num_conv)
        self.spatial_basic_list = []
        self.channel_reduce_list = []
        for scale in range(len(self.pts_channel_list)):
            self.pts_channel_list[scale] = self.pts_channel_list[scale] + 3
            self.spatial_basic = []
            for idx in range(num_conv - 1):
                self.spatial_basic.append(
                    nn.Conv2d(self.pts_channel_list[scale],
                            self.pts_channel_list[scale],
                            kernel_size=3,
                            stride=1,
                            padding=1))
                self.spatial_basic.append(
                    nn.BatchNorm2d(self.pts_channel_list[scale], eps=1e-3, momentum=0.01)
                )
                self.spatial_basic.append(
                    nn.ReLU()
                )
            self.spatial_basic.append(
                nn.Conv2d(self.pts_channel_list[scale], 1, kernel_size=3, stride=1, padding=1))
            self.spatial_basic_list.append(nn.Sequential(*self.spatial_basic))
            self.channel_reduce_list.append(nn.Conv2d(self.pts_channel_list[scale], self.img_channel_list[scale], kernel_size=1, stride=1))

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

                # Reverse the point cloud transformations to the original coords.
                if 'noise_scale' in batch_dict:
                    voxels_3d_batch[:, :3] /= batch_dict['noise_scale'][b]
                if 'noise_rot' in batch_dict:
                    voxels_3d_batch = common_utils.rotate_points_along_z(voxels_3d_batch[:, self.inv_idx].unsqueeze(0), -batch_dict['noise_rot'][b].unsqueeze(0))[0, :, self.inv_idx]
                if 'flip_x' in batch_dict:
                    voxels_3d_batch[:, 1] *= -1 if batch_dict['flip_x'][b] else 1
                if 'flip_y' in batch_dict:
                    voxels_3d_batch[:, 2] *= -1 if batch_dict['flip_y'][b] else 1

                ##chgd
                voxel_features_sparse = torch.cat([x_list[s].features[batch_index==b],voxels_3d_batch],dim=-1)

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
            new_img_feats.append(x_rgb[idx] + self.channel_reduce_list[idx].to(device=device)(attention_map * pts_img_list[idx]))

        return new_img_feats

class Patchv2(BasicGatev2):
    def __init__(self, img_channel_list, pts_channel_list, sparse_shape, voxel_size, point_cloud_range, inv_idx, pts_idx, num_conv = 2):
        super(Patchv2, self).__init__(img_channel_list, pts_channel_list, sparse_shape, voxel_size, point_cloud_range, inv_idx, num_conv)
        self.spatial_basic_list = []
        self.channel_reduce_list = []
        for scale in range(len(self.pts_channel_list)):
            self.pts_channel_list[scale] = self.pts_channel_list[scale] + 3
            self.spatial_basic = []
            for idx in range(num_conv - 1):
                self.spatial_basic.append(
                    nn.Conv2d(self.pts_channel_list[scale],
                            self.pts_channel_list[scale],
                            kernel_size=3,
                            stride=1,
                            padding=1))
                self.spatial_basic.append(
                    nn.BatchNorm2d(self.pts_channel_list[scale], eps=1e-3, momentum=0.01)
                )
                self.spatial_basic.append(
                    nn.ReLU()
                )
            self.spatial_basic.append(
                nn.Conv2d(self.pts_channel_list[scale], 1, kernel_size=3, stride=1, padding=1))
            self.spatial_basic_list.append(nn.Sequential(*self.spatial_basic))
            self.channel_reduce_list.append(nn.Conv2d(self.pts_channel_list[scale], self.img_channel_list[scale], kernel_size=1, stride=1))

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

                # Reverse the point cloud transformations to the original coords.
                if 'noise_scale' in batch_dict:
                    voxels_3d_batch[:, :3] /= batch_dict['noise_scale'][b]
                if 'noise_rot' in batch_dict:
                    voxels_3d_batch = common_utils.rotate_points_along_z(voxels_3d_batch[:, self.inv_idx].unsqueeze(0), -batch_dict['noise_rot'][b].unsqueeze(0))[0, :, self.inv_idx]
                if 'flip_x' in batch_dict:
                    voxels_3d_batch[:, 1] *= -1 if batch_dict['flip_x'][b] else 1
                if 'flip_y' in batch_dict:
                    voxels_3d_batch[:, 2] *= -1 if batch_dict['flip_y'][b] else 1
                    
                ##chgd
                voxel_features_sparse = torch.cat([x_list[s].features[batch_index==b],voxels_3d_batch],dim=-1)

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
            new_img_feats.append(x_rgb[idx]*attention_map)
            

        return new_img_feats

class BasicGatev4(nn.Module):
    def __init__(self, img_channel_list, pts_channel_list, sparse_shape, voxel_size, point_cloud_range, inv_idx, pts_idx, num_conv = 2):
        super(BasicGatev4, self).__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.inv_idx = inv_idx
        self.img_channel_list = img_channel_list
        self.pts_channel_list = [pts_channel_list[0]]
        self.sparse_shape = sparse_shape
        self.spatial_basic_list = []
        self.channel_reduce_list = []
        for scale in range(len(self.pts_channel_list)):
            self.spatial_basic = []
            for idx in range(num_conv - 1):
                self.spatial_basic.append(
                    nn.Conv2d(self.pts_channel_list[scale],
                            self.pts_channel_list[scale],
                            kernel_size=3,
                            stride=1,
                            padding=1))
                self.spatial_basic.append(
                    nn.BatchNorm2d(self.pts_channel_list[scale], eps=1e-3, momentum=0.01)
                )
                self.spatial_basic.append(
                    nn.ReLU()
                )
            self.spatial_basic.append(
                nn.Conv2d(self.pts_channel_list[scale], 1, kernel_size=3, stride=1, padding=1))
            self.spatial_basic_list.append(nn.Sequential(*self.spatial_basic))
            self.channel_reduce_list.append(nn.Conv2d(self.pts_channel_list[scale]+self.img_channel_list[scale], self.img_channel_list[scale], kernel_size=1, stride=1))
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
            cat_feat = torch.cat([x_rgb[idx],attention_map*pts_img_list[idx]],dim=1)
            new_img_feats.append(self.channel_reduce_list[idx].to(device=device)(cat_feat))

        return new_img_feats

class BasicGatev5(BasicGatev4):
    def __init__(self, img_channel_list, pts_channel_list, sparse_shape, voxel_size, point_cloud_range, inv_idx, pts_idx, num_conv = 2):
        super(BasicGatev5, self).__init__(img_channel_list, pts_channel_list, sparse_shape, voxel_size, point_cloud_range, inv_idx, num_conv)
        
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
            cat_feat = torch.cat([attention_map * x_rgb[idx],attention_map * pts_img_list[idx]],dim=1)
            new_img_feats.append(self.channel_reduce_list[idx].to(device=device)(cat_feat))

        return new_img_feats

class BasicGatev5_Patch(BasicGatev4):
    def __init__(self, img_channel_list, pts_channel_list, sparse_shape, voxel_size, point_cloud_range, inv_idx, pts_idx, num_conv = 2):
        super(BasicGatev5_Patch, self).__init__(img_channel_list, pts_channel_list, sparse_shape, voxel_size, point_cloud_range, inv_idx, num_conv)
        self.spatial_basic_list = []
        self.channel_reduce_list = []
        for scale in range(len(self.pts_channel_list)):
            self.pts_channel_list[scale] = self.pts_channel_list[scale] + 3
            self.spatial_basic = []
            for idx in range(num_conv - 1):
                self.spatial_basic.append(
                    nn.Conv2d(self.pts_channel_list[scale],
                            self.pts_channel_list[scale],
                            kernel_size=3,
                            stride=1,
                            padding=1))
                self.spatial_basic.append(
                    nn.BatchNorm2d(self.pts_channel_list[scale], eps=1e-3, momentum=0.01)
                )
                self.spatial_basic.append(
                    nn.ReLU()
                )
            self.spatial_basic.append(
                nn.Conv2d(self.pts_channel_list[scale], 1, kernel_size=3, stride=1, padding=1))
            self.spatial_basic_list.append(nn.Sequential(*self.spatial_basic))
            self.channel_reduce_list.append(nn.Conv2d(self.pts_channel_list[scale]+self.img_channel_list[scale], self.img_channel_list[scale], kernel_size=1, stride=1))

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
                voxel_features_sparse = torch.cat([x_list[s].features[batch_index==b],voxels_3d_batch],dim=-1)

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
            cat_feat = torch.cat([attention_map * x_rgb[idx],attention_map * pts_img_list[idx]],dim=1)
            new_img_feats.append(self.channel_reduce_list[idx].to(device=device)(cat_feat))

        return new_img_feats

class BasicGate_Patch(BasicGatev4):
    def __init__(self, img_channel_list, pts_channel_list, sparse_shape, voxel_size, point_cloud_range, inv_idx, pts_idx, num_conv = 2):
        super(BasicGate_Patch, self).__init__(img_channel_list, pts_channel_list, sparse_shape, voxel_size, point_cloud_range, inv_idx, num_conv)
        self.spatial_basic_list = []
        for scale in range(len(self.pts_channel_list)):
            self.pts_channel_list[scale] = self.pts_channel_list[scale] + 3
            self.spatial_basic = []
            for idx in range(num_conv - 1):
                self.spatial_basic.append(
                    nn.Conv2d(self.pts_channel_list[scale],
                            self.pts_channel_list[scale],
                            kernel_size=3,
                            stride=1,
                            padding=1))
                self.spatial_basic.append(
                    nn.BatchNorm2d(self.pts_channel_list[scale], eps=1e-3, momentum=0.01)
                )
                self.spatial_basic.append(
                    nn.ReLU()
                )
            self.spatial_basic.append(
                nn.Conv2d(self.pts_channel_list[scale], 1, kernel_size=3, stride=1, padding=1))
            self.spatial_basic_list.append(nn.Sequential(*self.spatial_basic))

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
                voxel_features_sparse = torch.cat([x_list[s].features[batch_index==b],voxels_3d_batch],dim=-1)

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
            new_img_feats.append(x_rgb[idx]*attention_map)
            

        return new_img_feats

class Basicgate_patch_iv_multivoxel(nn.Module):
    def __init__(self, img_channel_list, pts_channel_list, sparse_shape, voxel_size, point_cloud_range, inv_idx, pts_idx, num_conv = 2):
        super(Basicgate_patch_iv_multivoxel, self).__init__()
        self.voxel_idx = pts_idx
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.inv_idx = inv_idx
        self.img_channel_list = img_channel_list
        self.pts_channel_list = pts_channel_list
        
        self.sparse_shape = sparse_shape
        self.spatial_basic_list = []
        self.channel_reduce_list = []
        
        self.reduced_dim2=nn.Conv2d(self.pts_channel_list[self.voxel_idx[-1]] + 3,self.pts_channel_list[self.voxel_idx[-1]] + 3 ,kernel_size=1,stride=1,padding=0)
        self.reduced_dim3=nn.Conv2d(self.img_channel_list[0],1,kernel_size=1,stride=1,padding=0)

        self.spatial_basic = nn.Conv2d(self.pts_channel_list[self.voxel_idx[-1]]+3, 1, kernel_size=3, stride=1, padding=1)
        self.reduced_dim = []
        for conv_idx in range(self.voxel_idx[-1]):
            self.reduced_dim.append(
                nn.Conv2d(self.pts_channel_list[conv_idx] + 3,
                self.pts_channel_list[self.voxel_idx[-1]] + 3, kernel_size=1,stride=1,padding=0)
            )
        self.reduced_dim = nn.Sequential(*self.reduced_dim)

    def forward(self, x_rgb, x_list, batch_dict):
        batch_size = batch_dict['batch_size']
        calibs = batch_dict['calib']
        device = x_rgb[0].device
        img_shapes = [
            torch.tensor(f.shape[2:], device=device) for f in x_rgb
        ]
        h, w = batch_dict['images'].shape[2:]
        
        pts_img_list = []
        for conv_idx in range(len(self.pts_channel_list)):
            batch_img_list = []
            batch_index = x_list[conv_idx].indices[:, 0]
            ratio = self.sparse_shape[1] / x_list[conv_idx].spatial_shape[1]
            spatial_indices = x_list[conv_idx].indices[:, 1:] * ratio
            voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3]
            
            for b in range(batch_size):
                calib = calibs[b]
                voxels_3d_batch = voxels_3d[batch_index==b]
                voxel_features_sparse = x_list[conv_idx].features[batch_index==b]

                # Reverse the point cloud transformations to the original coords.
                if 'noise_scale' in batch_dict:
                    voxels_3d_batch[:, :3] /= batch_dict['noise_scale'][b]
                if 'noise_rot' in batch_dict:
                    voxels_3d_batch = common_utils.rotate_points_along_z(voxels_3d_batch[:, self.inv_idx].unsqueeze(0), -batch_dict['noise_rot'][b].unsqueeze(0))[0, :, self.inv_idx]
                if 'flip_x' in batch_dict:
                    voxels_3d_batch[:, 1] *= -1 if batch_dict['flip_x'][b] else 1
                if 'flip_y' in batch_dict:
                    voxels_3d_batch[:, 2] *= -1 if batch_dict['flip_y'][b] else 1
                voxel_features_sparse = torch.cat([x_list[conv_idx].features[batch_index==b],voxels_3d_batch],dim=-1)

                voxels_2d, _ = calib.lidar_to_img(voxels_3d_batch[:, self.inv_idx].cpu().numpy())
                voxels_2d_norm = voxels_2d / np.array([w, h])
                voxels_2d_norm_tensor = torch.Tensor(voxels_2d_norm).to(device)
                voxels_2d_norm_tensor = torch.clamp(voxels_2d_norm_tensor, min=0.0, max=1.0)
                pt_img, _ = pts2img(voxels_2d_norm_tensor, voxel_features_sparse, img_shapes[0], voxels_3d_batch)
                batch_img_list.append(pt_img.unsqueeze(0))
            pts_img_list.append(torch.cat(batch_img_list))
        
        for idx_ in self.voxel_idx:
            if idx_ != self.voxel_idx[-1]:
                pts2img_feat = self.reduced_dim[idx_].to(device=device)(pts_img_list[idx_])
            if idx_ == self.voxel_idx[0]:
                enhanced_pts_feat = pts2img_feat
            else:
                enhanced_pts_feat += pts_img_list[idx_]
        
        enhanced_pts_feat = self.reduced_dim2.to(device=device)(enhanced_pts_feat)
        gated_img_feat = self.reduced_dim3.to(device=device)(x_rgb[0])
        gated_img_feat = gated_img_feat.expand(batch_size,self.pts_channel_list[self.voxel_idx[-1]] + 3 , enhanced_pts_feat.shape[2],enhanced_pts_feat.shape[3])
        fused_feature = gated_img_feat+enhanced_pts_feat
        fused_feature = self.spatial_basic.to(device=device)(fused_feature)
        attention_map = torch.sigmoid(fused_feature)
        return [x_rgb[0] * attention_map]


class BasicGatev6(nn.Module):
    def __init__(self, img_channel_list, pts_channel_list, sparse_shape, voxel_size, point_cloud_range, inv_idx, pts_idx, num_conv = 2):
        super(BasicGatev6, self).__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.inv_idx = inv_idx
        self.img_channel_list = img_channel_list
        self.pts_channel_list = pts_channel_list
        self.sparse_shape = sparse_shape
        self.spatial_basic_list = []
        self.channel_reduce_list = []
        for scale in range(len(self.pts_channel_list)):
            self.spatial_basic = []
            for idx in range(num_conv - 1):
                self.spatial_basic.append(
                    nn.Conv2d(self.pts_channel_list[scale],
                            self.pts_channel_list[scale],
                            kernel_size=3,
                            stride=1,
                            padding=1))
                self.spatial_basic.append(
                    nn.BatchNorm2d(self.pts_channel_list[scale], eps=1e-3, momentum=0.01)
                )
                self.spatial_basic.append(
                    nn.ReLU()
                )
            self.spatial_basic.append(
                nn.Conv2d(self.pts_channel_list[scale], 1, kernel_size=3, stride=1, padding=1))
            self.spatial_basic_list.append(nn.Sequential(*self.spatial_basic))
            self.channel_reduce_list.append(nn.Conv2d(self.pts_channel_list[scale], self.img_channel_list[scale], kernel_size=1, stride=1))
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
            new_img_feats.append(x_rgb[idx] + self.channel_reduce_list[idx].to(device=device)(attention_map * pts_img_list[idx]))

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
    def visualize(i_pts_feat, i_coor):
        i_pts_feat = i_pts_feat[:,:,0]
        i_pts_feat[i_coor[:, 0], i_coor[:, 1]] = 1
        i_pts_feat = i_pts_feat.cpu().detach().numpy()*255
        cv2.imwrite("occupied_voxel.jpg", i_pts_feat)
    coor = coor[:, [1, 0]]
    i_shape = torch.cat(
        [shape + 1,
            torch.tensor([pts_feat.shape[1]]).cuda()])
    i_pts_feat = torch.zeros(tuple(i_shape), device=coor.device)
    i_coor = (coor * shape).to(torch.long)
    #visualize(i_pts_feat, i_coor)
    i_pts_feat[i_coor[:, 0], i_coor[:, 1]] = pts_feat
    i_pts_feat = i_pts_feat[:-1, :-1].permute(2, 0, 1)
    if ret_depth:
        i_shape[2] = 3
        i_depth_feat = torch.zeros(tuple(i_shape), device=coor.device)
        i_depth_feat[i_coor[:, 0], i_coor[:, 1]] = pts
        i_depth_feat = i_depth_feat[:-1, :-1].permute(2, 0, 1)
        return i_pts_feat, i_depth_feat
    
    return i_pts_feat, None



__all__ = {
    'devil': devil,
    'BasicGate': BasicGate,
    'BasicGatev2': BasicGatev2,
    'BasicGatev3': BasicGatev3,
    'BasicGatev4': BasicGatev4,
    'BasicGatev5': BasicGatev5,
    'BasicGatev6': BasicGatev6,
    'BasicGatev5_Patch': BasicGatev5_Patch,
    "BasicGate_Patch": BasicGate_Patch,
    "Basicgate_patch_iv_multivoxel": Basicgate_patch_iv_multivoxel,
    'BiGate': BiGate,
    'Patch': Patch,
    'Patchv2': Patchv2
}
