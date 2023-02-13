
import torch
import torch.nn as nn
import cv2
import numpy as np
from ..losses.auxseg_loss import SEGLOSS
import copy

class Basicgate_patch_iv_multivoxel(nn.Module):
    def __init__(self, **kwarg):
        super(Basicgate_patch_iv_multivoxel, self).__init__()
        self.img_num_channel = kwarg['img_num_channel']
        self.pts_num_channel = kwarg['pts_num_channel'] + 3
        self.voxel_feat_channel = kwarg['voxel_feat_channel']
        self.voxel_idx = kwarg['voxel_idx']
        if len(self.voxel_idx) == 1:
            self.pts_num_channel = self.voxel_feat_channel[self.voxel_idx[0]]
        
        self.reduced_dim2=nn.Conv2d(self.voxel_feat_channel[self.voxel_idx[-1]] + 3,self.voxel_feat_channel[self.voxel_idx[-1]] + 3 ,kernel_size=1,stride=1,padding=0)
        self.reduced_dim3=nn.Conv2d(self.img_num_channel,1,kernel_size=1,stride=1,padding=0)

        self.spatial_basic = nn.Conv2d(self.voxel_feat_channel[self.voxel_idx[-1]]+3, 1, kernel_size=3, stride=1, padding=1)
        self.reduced_dim = []
        for conv_idx in range(self.voxel_idx[-1]):
            self.reduced_dim.append(
                nn.Conv2d(self.voxel_feat_channel[conv_idx] + 3,
                self.voxel_feat_channel[self.voxel_idx[-1]] + 3, kernel_size=1,stride=1,padding=0)
            )
        self.reduced_dim = nn.Sequential(*self.reduced_dim)

    def forward(self, img_feat, voxel_feat, img_grid, voxel_coord,  batch_dict, cam_key, _idx, seg_prob):
        
        device = img_feat[0].device
        img_shapes = torch.tensor(img_feat.shape[1:], device = device)

        if len(self.voxel_idx) == 1:
            voxel_idx = self.voxel_idx[0]
            voxel_features = voxel_feat[voxel_idx]
            voxel_features = torch.cat([voxel_features,voxel_coord[voxel_idx]],dim=-1)
            pt_img = pts2img(img_grid[voxel_idx], voxel_features, img_shapes, batch_dict, cam_key, _idx, img_feat)
        else:
            for conv_idx in self.voxel_idx:
                voxel_features = voxel_feat[conv_idx]
                voxel_features = torch.cat([voxel_features,voxel_coord[conv_idx]],dim=-1)
                pts2img_feat = pts2img(img_grid[conv_idx], voxel_features, img_shapes, batch_dict, cam_key, _idx, img_feat)
                if conv_idx != self.voxel_idx[-1]:
                    pts2img_feat = self.reduced_dim[conv_idx].to(device=device)(pts2img_feat.unsqueeze(0).contiguous()).squeeze(0).contiguous()
                if conv_idx == self.voxel_idx[0]:
                    pt_img = pts2img_feat
                else:
                    pt_img += pts2img_feat

        #fused_feature = torch.cat([img_feat, pt_img], dim=0)
        pt_img = self.reduced_dim2.to(device=device)(pt_img.unsqueeze(0).contiguous()).squeeze(0).contiguous()
        gated_img_feat = self.reduced_dim3.to(device=device)(img_feat.unsqueeze(0).contiguous()).squeeze(0).contiguous()
        gated_img_feat = gated_img_feat.expand(self.voxel_feat_channel[self.voxel_idx[-1]] + 3 , pt_img.shape[1],pt_img.shape[2])
        #fused_feature = img_feat+pt_img
        fused_feature = gated_img_feat+pt_img
        fused_feature = self.spatial_basic.to(device=device)(fused_feature.unsqueeze(0).contiguous()).squeeze(0).contiguous()
        attention_map = torch.sigmoid(fused_feature)
        return img_feat * attention_map


class Basicgate_cvf(nn.Module):
    def __init__(self, **kwarg):
        super(Basicgate_cvf, self).__init__()
        self.img_num_channel = kwarg['img_num_channel']
        self.pts_num_channel = kwarg['pts_num_channel']
        self.voxel_feat_channel = kwarg['voxel_feat_channel']
        self.voxel_idx = kwarg['voxel_idx']
        if len(self.voxel_idx) == 1:
            self.pts_num_channel = self.voxel_feat_channel[self.voxel_idx[0]]
        self.spatial_basic = nn.Conv2d(self.pts_num_channel + self.img_num_channel, 1, kernel_size=3, stride=1, padding=1)
        self.reduced_dim = []
        for conv_idx in range(len(self.voxel_feat_channel)-1):
            self.reduced_dim.append(
                nn.Conv2d(self.voxel_feat_channel[conv_idx],
                self.voxel_feat_channel[-1], kernel_size=1,stride=1,padding=0)
            )
        self.reduced_dim = nn.Sequential(*self.reduced_dim)

    def forward(self, img_feat, voxel_feat, img_grid, voxel_coord,  batch_dict, cam_key, _idx, seg_prob):
        
        device = img_feat[0].device
        img_shapes = torch.tensor(img_feat.shape[1:], device = device)
        seg_prob = seg_prob[0][1].unsqueeze(0) # 0 : background, 1 : foreground
        enhanced_img_feat = img_feat * seg_prob

        if len(self.voxel_idx) == 1:
            voxel_idx = self.voxel_idx[0]
            voxel_features = voxel_feat[voxel_idx]
            pt_img = pts2img(img_grid[voxel_idx], voxel_features, img_shapes, batch_dict, cam_key, _idx, img_feat)
        else:
            for conv_idx in range(len(voxel_feat)):
                voxel_features = voxel_feat[conv_idx]
                pts2img_feat = pts2img(img_grid[conv_idx], voxel_features, img_shapes, batch_dict, cam_key, _idx, img_feat)
                if conv_idx != len(voxel_feat)-1:
                    pts2img_feat = self.reduced_dim[conv_idx].to(device=device)(pts2img_feat.unsqueeze(0).contiguous()).squeeze(0).contiguous()
                if conv_idx == 0:
                    pt_img = pts2img_feat
                else:
                    pt_img += pts2img_feat
        
        fused_feature = torch.cat([enhanced_img_feat, pt_img], dim=0)
        fused_feature = self.spatial_basic.to(device=device)(fused_feature.unsqueeze(0).contiguous()).squeeze(0).contiguous()
        attention_map = torch.sigmoid(fused_feature)
        return enhanced_img_feat * attention_map

class Foreground_fusion(nn.Module):
    def __init__(self, **kwarg):
        super(Foreground_fusion, self).__init__()
        self.img_num_channel = kwarg['img_num_channel']
        self.pts_num_channel = kwarg['pts_num_channel']
        self.voxel_feat_channel = kwarg['voxel_feat_channel']
        self.voxel_idx = kwarg['voxel_idx']
        if len(self.voxel_idx) == 1:
            self.pts_num_channel = self.voxel_feat_channel[self.voxel_idx[0]]
        self.spatial_basic = nn.Conv2d(self.pts_num_channel, self.img_num_channel, kernel_size=3, stride=1, padding=1)
        self.reduced_dim = []
        for conv_idx in range(len(self.voxel_feat_channel)-1):
            self.reduced_dim.append(
                nn.Conv2d(self.voxel_feat_channel[conv_idx],
                self.voxel_feat_channel[-1], kernel_size=1,stride=1,padding=0)
            )
        self.reduced_dim = nn.Sequential(*self.reduced_dim)
        self.conv2d = nn.Conv2d(self.img_num_channel,1,kernel_size=3, stride=1, padding=1)
    def forward(self, img_feat, voxel_feat, img_grid, voxel_coord,  batch_dict, cam_key, _idx, seg_prob):
        
        device = img_feat[0].device
        img_shapes = torch.tensor(img_feat.shape[1:], device = device)
        seg_prob = seg_prob[0][1].unsqueeze(0) # 0 : background, 1 : foreground
        seg_mask = (seg_prob > 0.5)
        masked_img_feat = img_feat * seg_mask
        
        if len(self.voxel_idx) == 1:
            voxel_idx = self.voxel_idx[0]
            voxel_features = voxel_feat[voxel_idx]
            pt_img = pts2img(img_grid[voxel_idx], voxel_features, img_shapes, batch_dict, cam_key, _idx, img_feat)
        else:
            for conv_idx in range(len(voxel_feat)):
                voxel_features = voxel_feat[conv_idx]
                pts2img_feat = pts2img(img_grid[conv_idx], voxel_features, img_shapes, batch_dict, cam_key, _idx, img_feat)
                if conv_idx != len(voxel_feat)-1:
                    pts2img_feat = self.reduced_dim[conv_idx].to(device=device)(pts2img_feat.unsqueeze(0).contiguous()).squeeze(0).contiguous()
                if conv_idx == 0:
                    pt_img = pts2img_feat
                else:
                    pt_img += pts2img_feat

        pt_img = self.spatial_basic.to(device=device)(pt_img.unsqueeze(0).contiguous()).squeeze(0).contiguous()
        masked_lidar_feat = pt_img * seg_mask
        masked_feat = masked_lidar_feat + masked_img_feat
        masked_feat = self.conv2d.to(device=device)(masked_feat.unsqueeze(0).contiguous()).squeeze(0).contiguous()
        attention_map = torch.sigmoid(masked_feat)
        
        return attention_map * img_feat



class Weighted_fusion(nn.Module):
    def __init__(self, **kwarg):
        super(Weighted_fusion, self).__init__()
        self.img_num_channel = kwarg['img_num_channel']
        self.pts_num_channel = kwarg['pts_num_channel']
        self.voxel_feat_channel = kwarg['voxel_feat_channel']
        self.voxel_idx = kwarg['voxel_idx']
        if len(self.voxel_idx) == 1:
            self.pts_num_channel = self.voxel_feat_channel[self.voxel_idx[0]]
        self.reduced_dim = []
        for conv_idx in range(len(self.voxel_feat_channel)-1):
            self.reduced_dim.append(
                nn.Conv2d(self.voxel_feat_channel[conv_idx],
                self.voxel_feat_channel[-1], kernel_size=1,stride=1,padding=0)
            )
        self.reduced_dim = nn.Sequential(*self.reduced_dim)
        self.channel_reduce = nn.Conv2d(self.pts_num_channel+self.img_num_channel,2,kernel_size=1,stride=1,padding=0)
        self.channel_reduce_ = nn.Conv2d(self.pts_num_channel+self.img_num_channel,self.img_num_channel,kernel_size=1,stride=1,padding=0)
    def forward(self, img_feat, voxel_feat, img_grid, voxel_coord,  batch_dict, cam_key, _idx, seg_prob):
        
        device = img_feat[0].device
        img_shapes = torch.tensor(img_feat.shape[1:], device = device)
        seg_prob = seg_prob[0][1].unsqueeze(0) # 0 : background, 1 : foreground
        enhanced_img_feat = img_feat * seg_prob

        if len(self.voxel_idx) == 1:
            voxel_idx = self.voxel_idx[0]
            voxel_features = voxel_feat[voxel_idx]
            pt_img = pts2img(img_grid[voxel_idx], voxel_features, img_shapes, batch_dict, cam_key, _idx, img_feat)
        else:
            for conv_idx in range(len(voxel_feat)):
                voxel_features = voxel_feat[conv_idx]
                pts2img_feat = pts2img(img_grid[conv_idx], voxel_features, img_shapes, batch_dict, cam_key, _idx, img_feat)
                if conv_idx != len(voxel_feat)-1:
                    pts2img_feat = self.reduced_dim[conv_idx].to(device=device)(pts2img_feat.unsqueeze(0).contiguous()).squeeze(0).contiguous()
                if conv_idx == 0:
                    pt_img = pts2img_feat
                else:
                    pt_img += pts2img_feat
        
        fused_feature = torch.cat([enhanced_img_feat, pt_img], dim=0)
        fused_feature = self.channel_reduce.to(device=device)(fused_feature.unsqueeze(0).contiguous()).squeeze(0).contiguous()
        attention_map = torch.sigmoid(fused_feature)
        enhanced_img_feat = attention_map[0].unsqueeze(0)*enhanced_img_feat
        enhanced_pts_feat = attention_map[1].unsqueeze(0)*pt_img
        enhanced_feat = torch.cat([enhanced_img_feat,enhanced_pts_feat],dim=0)
        enhanced_feat = self.channel_reduce_.to(device=device)(enhanced_feat.unsqueeze(0).contiguous()).squeeze(0).contiguous()
        return enhanced_feat

class Coord_Patched_Basicgate(nn.Module):
    def __init__(self, **kwarg):
        super(Coord_Patched_Basicgate, self).__init__()
        self.img_num_channel = kwarg['img_num_channel']
        self.pts_num_channel = kwarg['pts_num_channel']
        self.pts_num_channel = self.pts_num_channel + 3
        self.spatial_basic = []
        self.spatial_basic.append(
            nn.Conv2d(self.pts_num_channel,
                    self.pts_num_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1))
        self.spatial_basic.append(
            nn.BatchNorm2d(self.pts_num_channel, eps=1e-3, momentum=0.01)
        )
        self.spatial_basic.append(
            nn.ReLU()
        )
        self.spatial_basic.append(
            nn.Conv2d(self.pts_num_channel, 1, kernel_size=3, stride=1, padding=1))
        self.spatial_basic = nn.Sequential(*self.spatial_basic)

    def forward(self, img_feat, voxel_feat, img_grid, voxel_coord):
        
        device = img_feat[0].device
        
        img_shapes = torch.tensor(img_feat.shape[1:], device = device)
        voxel_features = torch.cat([voxel_feat,voxel_coord],dim=-1)
        pt_img = pts2img(img_grid, voxel_features, img_shapes)
        pts = self.spatial_basic.to(device=device)(pt_img.unsqueeze(0).contiguous())
        attention_map = torch.sigmoid(pts.squeeze(0).contiguous())
        return img_feat*attention_map

class BasicGate(nn.Module):
    def __init__(self, **kwarg):
        super(BasicGate, self).__init__()
        self.img_num_channel = kwarg['img_num_channel']
        self.pts_num_channel = kwarg['pts_num_channel']
        self.voxel_feat_channel = kwarg['voxel_feat_channel']
        self.voxel_idx = kwarg['voxel_idx']
        if len(self.voxel_idx) == 1:
            self.pts_num_channel = self.voxel_feat_channel[self.voxel_idx[0]]
        self.spatial_basic = []
        self.spatial_basic.append(
            nn.Conv2d(self.pts_num_channel,
                    self.pts_num_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1))
        self.spatial_basic.append(
            nn.BatchNorm2d(self.pts_num_channel, eps=1e-3, momentum=0.01)
        )
        self.spatial_basic.append(
            nn.ReLU()
        )
        self.spatial_basic.append(
            nn.Conv2d(self.pts_num_channel, 1, kernel_size=3, stride=1, padding=1))
        self.spatial_basic = nn.Sequential(*self.spatial_basic)
        self.reduced_dim = []
        for conv_idx in range(len(self.voxel_feat_channel)-1):
            self.reduced_dim.append(
                nn.Conv2d(self.voxel_feat_channel[conv_idx],
                self.voxel_feat_channel[-1], kernel_size=1,stride=1,padding=0)
            )
        self.reduced_dim = nn.Sequential(*self.reduced_dim)

    def forward(self, img_feat, voxel_feat, img_grid, voxel_coord,  batch_dict, cam_key, _idx):
        
        device = img_feat[0].device
        img_shapes = torch.tensor(img_feat.shape[1:], device = device)
        if len(self.voxel_idx) == 1:
            voxel_idx = self.voxel_idx[0]
            voxel_features = voxel_feat[voxel_idx]
            pt_img = pts2img(img_grid[voxel_idx], voxel_features, img_shapes, batch_dict, cam_key, _idx, img_feat)
        else:
            for conv_idx in range(len(voxel_feat)):
                voxel_features = voxel_feat[conv_idx]
                pts2img_feat = pts2img(img_grid[conv_idx], voxel_features, img_shapes, batch_dict, cam_key, _idx, img_feat)
                if conv_idx != len(voxel_feat)-1:
                    pts2img_feat = self.reduced_dim[conv_idx].to(device=device)(pts2img_feat.unsqueeze(0).contiguous()).squeeze(0).contiguous()
                if conv_idx == 0:
                    pt_img = pts2img_feat
                else:
                    pt_img += pts2img_feat
        pts = self.spatial_basic.to(device=device)(pt_img.unsqueeze(0).contiguous())
        attention_map = torch.sigmoid(pts.squeeze(0).contiguous())
        return img_feat*attention_map

class BasicGatev2(nn.Module):
    def __init__(self, **kwarg):
        super(BasicGatev2, self).__init__()
        self.img_num_channel = kwarg['img_num_channel']
        self.pts_num_channel = kwarg['pts_num_channel']
        self.spatial_basic = []
        self.spatial_basic.append(
            nn.Conv2d(self.pts_num_channel,
                    self.pts_num_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1))
        self.spatial_basic.append(
            nn.BatchNorm2d(self.pts_num_channel, eps=1e-3, momentum=0.01)
        )
        self.spatial_basic.append(
            nn.ReLU()
        )
        self.spatial_basic.append(
            nn.Conv2d(self.pts_num_channel, 1, kernel_size=3, stride=1, padding=1))
        self.spatial_basic = nn.Sequential(*self.spatial_basic)
        self.channel_reduce_list = nn.Conv2d(self.pts_num_channel,self.img_num_channel,kernel_size=1,stride=1)

    def forward(self, img_feat, voxel_feat, img_grid, voxel_coord):
        
        device = img_feat[0].device
        
        img_shapes = torch.tensor(img_feat.shape[1:], device = device)
        voxel_features = voxel_feat
        pt_img = pts2img(img_grid, voxel_features, img_shapes)
        pts = self.spatial_basic.to(device=device)(pt_img.unsqueeze(0).contiguous())
        attention_map = torch.sigmoid(pts.squeeze(0).contiguous())
        
        return img_feat + self.channel_reduce_list.to(device=device)((attention_map*pt_img).unsqueeze(0)).squeeze(0)

class BasicGatev3(BasicGatev2):
    def __init__(self, **kwarg):
        super(BasicGatev3, self).__init__(**kwarg)
        

    def forward(self, img_feat, voxel_feat, img_grid, voxel_coord):
        
        device = img_feat[0].device
        
        img_shapes = torch.tensor(img_feat.shape[1:], device = device)
        voxel_features = voxel_feat
        pt_img = pts2img(img_grid, voxel_features, img_shapes)
        pts = self.spatial_basic.to(device=device)(pt_img.unsqueeze(0).contiguous())
        attention_map = torch.sigmoid(pts.squeeze(0).contiguous())
        
        return img_feat*attention_map + self.channel_reduce_list.to(device=device)((attention_map*pt_img).unsqueeze(0)).squeeze(0)

class BasicGatev4(BasicGatev2):
    def __init__(self, **kwarg):
        super(BasicGatev4, self).__init__(**kwarg)
        self.channel_reduce_list = nn.Conv2d(self.pts_num_channel+self.img_num_channel,self.img_num_channel,kernel_size=1,stride=1)

    def forward(self, img_feat, voxel_feat, img_grid, voxel_coord):
        
        device = img_feat[0].device
        
        img_shapes = torch.tensor(img_feat.shape[1:], device = device)
        voxel_features = voxel_feat
        pt_img = pts2img(img_grid, voxel_features, img_shapes)
        pts = self.spatial_basic.to(device=device)(pt_img.unsqueeze(0).contiguous())
        attention_map = torch.sigmoid(pts.squeeze(0).contiguous())
        cat_feat = torch.cat([img_feat,attention_map*pt_img],dim=1)

        
        return self.channel_reduce_list.to(device=device)(cat_feat)

class BasicGatev5(BasicGatev2):
    def __init__(self, **kwarg):
        super(BasicGatev5, self).__init__(**kwarg)
        self.channel_reduce_list = nn.Conv2d(self.pts_num_channel+self.img_num_channel,self.img_num_channel,kernel_size=1,stride=1)

    def forward(self, img_feat, voxel_feat, img_grid, voxel_coord):
        
        device = img_feat[0].device
        
        img_shapes = torch.tensor(img_feat.shape[1:], device = device)
        voxel_features = voxel_feat
        pt_img = pts2img(img_grid, voxel_features, img_shapes)
        pts = self.spatial_basic.to(device=device)(pt_img.unsqueeze(0).contiguous())
        attention_map = torch.sigmoid(pts.squeeze(0).contiguous())
        cat_feat = torch.cat([attention_map*img_feat,attention_map*pt_img],dim=1)        
        return self.channel_reduce_list.to(device=device)(cat_feat)

class Coord_Patched_Basicgatev5(BasicGatev2):
    def __init__(self, **kwarg):
        kwarg['pts_num_channel'] = kwarg['pts_num_channel'] + 3
        super(Coord_Patched_Basicgatev5, self).__init__(**kwarg)
        self.channel_reduce_list = nn.Conv2d(self.pts_num_channel+self.img_num_channel,self.img_num_channel,kernel_size=1,stride=1)

    def forward(self, img_feat, voxel_feat, img_grid, voxel_coord):
        
        device = img_feat[0].device
        
        img_shapes = torch.tensor(img_feat.shape[1:], device = device)
        voxel_features = torch.cat([voxel_feat,voxel_coord],dim=-1)
        pt_img = pts2img(img_grid, voxel_features, img_shapes)
        pts = self.spatial_basic.to(device=device)(pt_img.unsqueeze(0).contiguous())
        attention_map = torch.sigmoid(pts.squeeze(0).contiguous())
        cat_feat = torch.cat([attention_map*img_feat,attention_map*pt_img],dim=1)        
        return self.channel_reduce_list.to(device=device)(cat_feat)
        
class Coord_Patched_Basicgatev2(BasicGatev2):
    def __init__(self, **kwarg):
        kwarg['pts_num_channel'] = kwarg['pts_num_channel'] + 3
        super(Coord_Patched_Basicgatev2, self).__init__(**kwarg)

    def forward(self, img_feat, voxel_feat, img_grid, voxel_coord):
        
        device = img_feat[0].device
        
        img_shapes = torch.tensor(img_feat.shape[1:], device = device)
        voxel_features = torch.cat([voxel_feat,voxel_coord],dim=-1)
        pt_img = pts2img(img_grid, voxel_features, img_shapes)
        pts = self.spatial_basic.to(device=device)(pt_img.unsqueeze(0).contiguous())
        attention_map = torch.sigmoid(pts.squeeze(0).contiguous())
        
        return img_feat + self.channel_reduce_list.to(device=device)((attention_map*pt_img).unsqueeze(0)).squeeze(0)


def pts2img(coor, pts_feat, shape, batch_dict, cam_key, _idx, img_feat):
    def visualize_img(batch_dict,cam_key):
        # pts_feat = pts_feat.detach().cpu().max(2)[0].numpy()
        # pts_feat = (pts_feat * 255.).astype(np.uint8)
        # cv2.imwrite('lidar2img.png', pts_feat)
        cv2.imwrite("aaa.jpg",batch_dict['images'][cam_key][0].detach().cpu().numpy()*255)

    def visualize_feat(img_feat):
        feat = img_feat.cpu().detach().numpy()
        min = feat.min()
        max = feat.max()
        image_features = (feat-min)/(max-min)
        image_features = (image_features*255)
        max_image_feature = np.max(np.transpose(image_features.astype("uint8"),(1,2,0)),axis=2)
        max_image_feature = cv2.applyColorMap(max_image_feature,cv2.COLORMAP_JET)
        cv2.imwrite("max_image_feature.jpg",max_image_feature)
        return max_image_feature
    def visualize_voxels(i_coor,max_img_feat):
        for coor in i_coor:
            cv2.circle(max_img_feat,(coor[1].item(),coor[0].item()), 1,(0,0,255))
        cv2.imwrite("pointed_feature.jpg",max_img_feat)
    def visualize_overlaped_voxels(i_coor, max_img_feat):
        overlaped_voxel_mask = torch.unique(i_coor,dim=0,return_counts=True)[1]>1
        voxel_after_remove = torch.unique(i_coor,dim=0,return_counts=True)[0]
        overlaped_voxel = voxel_after_remove[overlaped_voxel_mask]
        for coor in overlaped_voxel:
            cv2.circle(max_img_feat,(coor[1].item(),coor[0].item()), 1,(0,0,255))
        cv2.imwrite("overlaped_pointed_feature.jpg",max_img_feat)


    coor = coor[:, [1, 0]] #H,W
    i_shape = torch.cat(
        [shape + 1,
            torch.tensor([pts_feat.shape[1]]).cuda()])
    i_pts_feat = torch.zeros(tuple(i_shape), device=coor.device)
    #i_coor = (coor * shape).to(torch.long)
    i_coor = coor.to(torch.long)
    i_pts_feat[i_coor[:, 0], i_coor[:, 1]] = pts_feat
    i_pts_feat = i_pts_feat[:-1, :-1].permute(2, 0, 1)

    #visualize_img(batch_dict, cam_key)
    #max_img_feat = visualize_feat(img_feat)
    #visualize_voxels(i_coor,copy.deepcopy(max_img_feat))
    #visualize_overlaped_voxels(i_coor,copy.deepcopy(max_img_feat))
    
    
    return i_pts_feat



__all__ = {
    "Basicgate_cvf": Basicgate_cvf,
    "Foreground_fusion": Foreground_fusion,
    "Weighted_fusion": Weighted_fusion,
    "Basicgate_patch_iv_multivoxel": Basicgate_patch_iv_multivoxel,
    "Coord_Patched_Basicgate": Coord_Patched_Basicgate,
    "BasicGate": BasicGate,
    "BasicGatev2": BasicGatev2,
    "BasicGatev3": BasicGatev3,
    "BasicGatev4": BasicGatev4,
    "BasicGatev5": BasicGatev5,
    "Coord_Patched_Basicgatev2": Coord_Patched_Basicgatev2,
    "Coord_Patched_Basicgatev5": Coord_Patched_Basicgatev5
}
