import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.models.registry import FUSION
from det3d.models.model_utils.basic_block_1d import BasicBlock1D
from det3d.models.model_utils.actr import build as build_actr
from .point_to_image_projection import Point2ImageProjection
from det3d.models.model_utils import attention, segloss
from det3d.core.bbox import box_np_ops
import numpy as np
from det3d.datasets.nuscenes.nusc_common import get_lidar2cam_matrix, view_points

@FUSION.register_module
class VoxelWithPointProjection(nn.Module):
    def __init__(self, fuse_mode, interpolate, voxel_size, pc_range, image_list, image_scale=1,
                 depth_thres=0, double_flip=False, layer_channel=None, pfat_cfg=None, lt_cfg=None,
                 ifat_cfg=None, seg_cfg=None, model_name='ACTR'
                 ):
        """
        Initializes module to transform frustum features to voxel features via 3D transformation and sampling
        Args:
            voxel_size: [X, Y, Z], Voxel grid size
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
        """
        super().__init__()
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.point_projector = Point2ImageProjection(voxel_size=voxel_size,
                                                     pc_range=pc_range,
                                                     depth_thres=depth_thres,
                                                     double_flip=double_flip)
        self.fuse_mode = fuse_mode
        self.image_interp = interpolate
        self.image_list = image_list
        self.image_scale = image_scale
        self.double_flip = double_flip
        if self.fuse_mode == 'concat':
            self.fuse_blocks = nn.ModuleDict()
            for _layer in layer_channel.keys():
                block_cfg = {"in_channels": layer_channel[_layer]*2,
                             "out_channels": layer_channel[_layer],
                             "kernel_size": 1,
                             "stride": 1,
                             "bias": False}
                self.fuse_blocks[_layer] = BasicBlock1D(**block_cfg)
        if self.fuse_mode == 'pfat':
            self.pfat = build_actr(pfat_cfg, lt_cfg=lt_cfg, model_name=model_name)

        self.ifat_cfg = None
        if ifat_cfg:
            self.ifat_cfg = ifat_cfg
            self.ifat = attention.__all__[self.ifat_cfg.fusion_method](
                **self.ifat_cfg
            )

        self.seg_cfg = None
        if seg_cfg:
            self.seg_cfg = seg_cfg
            self.seg = segloss.__all__[self.seg_cfg.seg_method](
                **self.seg_cfg
            )


    def fusion(self, image_feat, voxel_feat, image_grid, layer_name=None, point_inv=None, fuse_mode=None):
        """
        Fuses voxel features and image features
        Args:
            image_feat: (C, H, W), Encoded image features
            voxel_feat: (N, C), Encoded voxel features
            image_grid: (N, 2), Image coordinates in X,Y of image plane
        Returns:
            voxel_feat: (N, C), Fused voxel features
        """
        image_grid = image_grid[:,[1,0]] # X,Y -> Y,X

        if fuse_mode == 'sum':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            voxel_feat += fuse_feat.permute(1,0)
        elif fuse_mode == 'mean':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            voxel_feat = (voxel_feat + fuse_feat.permute(1,0)) / 2
        elif fuse_mode == 'concat':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            concat_feat = torch.cat([fuse_feat, voxel_feat.permute(1,0)], dim=0)
            voxel_feat = self.fuse_blocks[layer_name](concat_feat.unsqueeze(0))[0]
            voxel_feat = voxel_feat.permute(1,0)
        else:
            raise NotImplementedError
        
        return voxel_feat

    def generate_2D_GT(self, cam_key, _idx, example, data_dict):
        
        cur_cam_key = cam_key[4:]
        cur_batch = _idx
        lidar2cam = example['calib']["lidar2cam_"+ cur_cam_key][cur_batch]
        cam_intrinsic = example['calib']["cam_intrinsic_"+ cur_cam_key][cur_batch]
        cur_image = example['cam'][cam_key][cur_batch]
        gt_boxes3d = example['gt_boxes_noaug'][cur_batch]
        image_scale = example['image_scale'][cur_batch]

        sample_coords = box_np_ops.rbbox3d_to_corners(gt_boxes3d[:,:9])
        points_3d = np.concatenate([sample_coords, np.ones((*sample_coords.shape[:2], 1))], axis=-1)
        lidar2cam = lidar2cam.cpu().numpy()
        cam_intrinsic = cam_intrinsic.cpu().numpy()
        image_scale = image_scale
        points_cam = (points_3d @ lidar2cam.T).T
        cam_mask = (points_cam[2] > 0).all(axis=0)
        points_cam = points_cam[...,cam_mask].reshape(4, -1)
        points_cam = points_cam.reshape(4, -1)
        point_img = view_points(points_cam[:3, :], np.array(cam_intrinsic), normalize=True)
        point_img = point_img.reshape(3, 8, -1)
        point_img = point_img.transpose()[...,:2]
        minxy = np.min(point_img, axis=-2)
        maxxy = np.max(point_img, axis=-2)
        bbox = np.concatenate([minxy, maxxy], axis=-1)
        bbox = (bbox * image_scale).astype(np.int32)
        bbox[:,0::2] = np.clip(bbox[:,0::2], a_min=0, a_max=cur_image.shape[1]-1)
        bbox[:,1::2] = np.clip(bbox[:,1::2], a_min=0, a_max=cur_image.shape[0]-1)
        cam_mask = (bbox[:,2]-bbox[:,0])*(bbox[:,3]-bbox[:,1])>0
        bbox = bbox[cam_mask]

        data_dict.update({"gt_boxes2d": bbox})
        # visualize
        #cur_image = cur_image.numpy()*255
        # for bbx in bbox:
        #     cv2.rectangle(cur_image, (int(bbx[0]), int(bbx[1])), (int(bbx[2]),int(bbx[3])) ,(255,0,0),2)
        # cv2.imwrite('demo.png',cur_image)        
        return data_dict

    def forward(self, batch_dict, example, encoded_voxel_list=None, layer_name=None, img_conv_func=None, fuse_mode=None, d_factor_list=None):
        """
        Generates voxel features via 3D transformation and sampling
        Args:
            batch_dict:
                voxel_coords: (N, 4), Voxel coordinates with B,Z,Y,X
                lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
                cam_to_img: (B, 3, 4), Camera projection matrix
                image_shape: (B, 2), Image shape [H, W]
            encoded_voxel: (N, C), Sparse Voxel featuress
        Returns:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Image voxel features
            voxel_features: (N, C), Sparse Image voxel features
    """
        batch_size = len(batch_dict['image_shape'][self.image_list[0].lower()])
        if fuse_mode == 'pfat':
            img_feat_n = []
            img_feat_n_ms = []
            img_grid_n = [[] for _ in range(batch_size)]
            v_feat_n = [[] for _ in range(batch_size)]
            v_i_feat_n = [[] for _ in range(batch_size)]
            point_inv_n = [[] for _ in range(batch_size)]
            mask_n = [[] for _ in range(batch_size)]
            auxloss_n = [[] for _ in range(batch_size)]
        encoded_voxel = encoded_voxel_list[-1]
        for cam_key in self.image_list:
            cam_key = cam_key.lower()
            # Generate sampling grid for frustum volume
            projection_dict_list = []
            for conv_idx in range(len(encoded_voxel_list)):
                projection_dict = self.point_projector(voxel_coords=encoded_voxel_list[conv_idx].indices.float(),
                                                       image_scale=self.image_scale,
                                                       batch_dict=batch_dict, 
                                                       cam_key=cam_key,
                                                       d_factor=d_factor_list[conv_idx]
                                                       )
                projection_dict_list.append(projection_dict)

            # check 
            if encoded_voxel is not None:
                in_bcakbone = True
            else:
                in_bcakbone = False
                encoded_voxel = batch_dict['encoded_spconv_tensor']
            if not self.training and self.double_flip:
                batch_size = batch_size * 4
            for _idx in range(batch_size): #(len(batch_dict['image_shape'][cam_key])):
                _idx_key = _idx//4 if self.double_flip else _idx
                image_feat = batch_dict['img_feat'][layer_name+'_feat2d'][cam_key][_idx_key]
                if img_conv_func:
                    image_feat = img_conv_func(image_feat.unsqueeze(0))[0]
                raw_shape = tuple(batch_dict['image_shape'][cam_key][_idx_key].cpu().numpy())
                feat_shape = image_feat.shape[-2:]
                if self.image_interp:
                    image_feat = F.interpolate(image_feat.unsqueeze(0), size=raw_shape[:2], mode='bilinear')[0]

                """
                index_mask = encoded_voxel.indices[:,0]==_idx
                voxel_feat = encoded_voxel.features[index_mask]
                image_grid = projection_dict['image_grid'][_idx]
                voxel_grid = projection_dict['batch_voxel'][_idx]
                point_mask = projection_dict['point_mask'][_idx]
                image_depth = projection_dict['image_depths'][_idx]
                point_inv = projection_dict['point_inv'][_idx]
                # temporary use for validation
                # point_mask[len(voxel_feat):] -> 0 for batch construction
                voxel_mask = point_mask[:len(voxel_feat)]
                if self.training and 'overlap_mask' in batch_dict.keys():
                    overlap_mask = batch_dict['overlap_mask'][_idx]
                    is_overlap = overlap_mask[image_grid[:,1], image_grid[:,0]].bool()
                    if 'depth_mask' in batch_dict.keys():
                        depth_mask = batch_dict['depth_mask'][_idx]
                        depth_range = depth_mask[image_grid[:,1], image_grid[:,0]]
                        is_inrange = (image_depth > depth_range[:,0]) & (image_depth < depth_range[:,1])
                        is_overlap = is_overlap & (~is_inrange)

                    image_grid = image_grid[~is_overlap]
                    voxel_grid = voxel_grid[~is_overlap]
                    point_mask = point_mask[~is_overlap]
                    point_inv = point_inv[~is_overlap]
                    voxel_mask = voxel_mask & (~is_overlap[:len(voxel_feat)])
                if not self.image_interp:
                    image_grid = image_grid.float()
                    image_grid[:,0] *= (feat_shape[1]/raw_shape[1])
                    image_grid[:,1] *= (feat_shape[0]/raw_shape[0])
                    image_grid = image_grid.long()

                if fuse_mode == 'pfat':
                    img_feat_n.append(image_feat)
                    img_grid_n[_idx].append(image_grid[point_mask])
                    point_inv_n[_idx].append(point_inv[point_mask])
                    v_feat_n[_idx].append(voxel_feat[voxel_mask])
                    mask_n[_idx].append(voxel_mask)
                    # for image feat
                    v_i_feat = image_feat[:, image_grid[point_mask][:, 1], image_grid[point_mask][:, 0]]
                    v_i_feat_n[_idx].append(v_i_feat.permute(1, 0))
                else:
                    voxel_feat[voxel_mask] = self.fusion(image_feat, voxel_feat[voxel_mask], 
                                                         image_grid[point_mask], layer_name, 
                                                         point_inv[point_mask], fuse_mode=fuse_mode)
                    encoded_voxel.features[index_mask] = voxel_feat
                """
                voxel_feat_list=[]
                voxel_mask_list=[]
                image_grid_list=[]
                point_mask_list=[]
                point_inv_list=[]
                for _conv_idx in range(len(encoded_voxel_list)):
                    index_mask = encoded_voxel_list[_conv_idx].indices[:,0]==_idx
                    voxel_feat = encoded_voxel_list[_conv_idx].features[index_mask]
                    image_grid = projection_dict_list[_conv_idx]['image_grid'][_idx]
                    voxel_grid = projection_dict_list[_conv_idx]['batch_voxel'][_idx]
                    point_mask = projection_dict_list[_conv_idx]['point_mask'][_idx]
                    image_depth = projection_dict_list[_conv_idx]['image_depths'][_idx]
                    point_inv = projection_dict_list[_conv_idx]['point_inv'][_idx]
                    # temporary use for validation
                    # point_mask[len(voxel_feat):] -> 0 for batch construction
                    voxel_mask = point_mask[:len(voxel_feat)]
                    if self.training and 'overlap_mask' in batch_dict.keys():
                        overlap_mask = batch_dict['overlap_mask'][_idx]
                        is_overlap = overlap_mask[image_grid[:,1], image_grid[:,0]].bool()
                        if 'depth_mask' in batch_dict.keys():
                            depth_mask = batch_dict['depth_mask'][_idx]
                            depth_range = depth_mask[image_grid[:,1], image_grid[:,0]]
                            is_inrange = (image_depth > depth_range[:,0]) & (image_depth < depth_range[:,1])
                            is_overlap = is_overlap & (~is_inrange)

                        image_grid = image_grid[~is_overlap]
                        voxel_grid = voxel_grid[~is_overlap]
                        point_mask = point_mask[~is_overlap]
                        point_inv = point_inv[~is_overlap]
                        voxel_mask = voxel_mask & (~is_overlap[:len(voxel_feat)])
                    if not self.image_interp:
                        image_grid = image_grid.float()
                        image_grid[:,0] *= (feat_shape[1]/raw_shape[1])
                        image_grid[:,1] *= (feat_shape[0]/raw_shape[0])
                        image_grid = image_grid.long()
                    voxel_feat_list.append(voxel_feat)
                    voxel_mask_list.append(voxel_mask)
                    image_grid_list.append(image_grid)
                    point_mask_list.append(point_mask)
                    point_inv_list.append(point_inv)


                if fuse_mode == 'pfat':
                    if self.ifat_cfg is not None:
                        if self.seg_cfg is not None:
                            if 'gt_boxes_noaug' in example:
                                batch_dict = self.generate_2D_GT(cam_key, _idx, example, batch_dict)
                            auxloss_list, seg_prob = self.seg(image_feat, batch_dict, cam_key, _idx)
                            if 'gt_boxes_noaug' in example:    
                                auxloss_n[_idx].append(auxloss_list)
                        else:
                            seg_prob = None
                        image_feat = self.ifat(image_feat,
                                               [voxel_feat_list[idx][voxel_mask_list[idx]] for idx in range(len(voxel_feat_list))], 
                                               [image_grid_list[idx][point_mask_list[idx]] for idx in range(len(voxel_feat_list))],
                                               [point_inv_list[idx][point_mask_list[idx]] for idx in range(len(voxel_feat_list))],
                                               batch_dict, 
                                               cam_key, 
                                               _idx,
                                               seg_prob)
                    img_feat_n.append(image_feat)
                    img_grid_n[_idx].append(image_grid[point_mask])
                    point_inv_n[_idx].append(point_inv[point_mask])
                    v_feat_n[_idx].append(voxel_feat[voxel_mask])
                    mask_n[_idx].append(voxel_mask)
                    # for image feat
                    v_i_feat = image_feat[:, image_grid[point_mask][:, 1], image_grid[point_mask][:, 0]]
                    v_i_feat_n[_idx].append(v_i_feat.permute(1, 0))
                else:
                    voxel_feat[voxel_mask] = self.fusion(image_feat, voxel_feat[voxel_mask], 
                                                         image_grid[point_mask], layer_name, 
                                                         point_inv[point_mask], fuse_mode=fuse_mode)
                    encoded_voxel.features[index_mask] = voxel_feat

        if 'layer2_ori_feat2d' in batch_dict['img_feat']:
            for cam_key in self.image_list:
                cam_key = cam_key.lower()
                for _idx in range(batch_size): #(len(batch_dict['image_shape'][cam_key])):
                    img_feat_n_ms.append(batch_dict['img_feat']['layer2_ori_feat2d'][cam_key][_idx_key])
        if fuse_mode == 'pfat':
            # 6*b, c, w, h -> b*6, c, w, h
            img_pfat = []
            img_feat_n = torch.stack(img_feat_n)
            c, w, h = img_feat_n.shape[1:]
            img_feat_n = img_feat_n.reshape(6, batch_size, c, w, h)
            img_feat_n = img_feat_n.transpose(1, 0).reshape(-1, c, w, h)
            img_pfat.append(img_feat_n)
            if len(img_feat_n_ms):
                img_feat_n_ms = torch.stack(img_feat_n_ms)
                c2, w2, h2 = img_feat_n_ms.shape[1:]
                img_feat_n_ms = img_feat_n_ms.reshape(6, batch_size, c2, w2, h2)
                img_feat_n_ms = img_feat_n_ms.transpose(1, 0).reshape(-1, c2, w2, h2)
                img_pfat.append(img_feat_n_ms)


            # aggregate
            max_ne = max([img_grid_n[b][i].shape[0] for b in range(batch_size) for i in range(6)])
            v_channel = voxel_feat.shape[1]
            img_grid_b = torch.zeros((batch_size*6, max_ne, 2)).cuda()
            pts_inv_b = torch.zeros((batch_size*6, max_ne, 3)).cuda()
            v_feat_b = torch.zeros((batch_size*6, max_ne, v_channel)).cuda()
            v_i_feat_b = torch.zeros((batch_size*6, max_ne, c)).cuda()
            for b in range(batch_size):
                for i in range(6):
                    ne = img_grid_n[b][i].shape[0]
                    img_grid_b[b*6+i, :ne] = img_grid_n[b][i]
                    pts_inv_b[b*6+i, :ne] = point_inv_n[b][i]
                    v_feat_b[b*6+i, :ne] = v_feat_n[b][i]
                    v_i_feat_b[b*6+i, :ne] = v_i_feat_n[b][i]

            # for visualize
            if False:
                import cv2
                cam_list = list(batch_dict['images'].keys())
                for b in range(batch_size):
                    for i in range(6):
                        img_feat = img_feat_n[b*6+i].max(0)[0].detach().cpu()
                        img_feat_norm = ((img_feat - img_feat.min()) / (img_feat.max() - img_feat.min()) * 255).to(torch.uint8).numpy()
                        img_feat_norm_jet = cv2.applyColorMap(img_feat_norm, cv2.COLORMAP_JET)
                        cv2.imwrite('./vis/%06d_feat.png' % (b*6+i), img_feat_norm_jet)

                        img_grid = img_grid_b[b*6+i].to(torch.uint8).detach().cpu().numpy()
                        img_feat_norm[img_grid[:, 1], img_grid[:, 0]] = 255
                        img_feat_norm_jet = cv2.applyColorMap(img_feat_norm, cv2.COLORMAP_JET)
                        cv2.imwrite('./vis/%06d_feat_proj.png' % (b*6+i), img_feat_norm_jet)

                        img_o = batch_dict['images'][cam_list[i]][b]
                        img_o = ((img_o - img_o.min()) / (img_o.max() - img_o.min()) * 255).to(torch.uint8).numpy()
                        cv2.imwrite('./vis/%06d_original.png' % (b*6+i), img_o)

            img_grid_b /= torch.tensor(feat_shape[::-1]).cuda()
            enh_feat = self.pfat(v_feat=v_feat_b, grid=img_grid_b, i_feats=img_pfat, 
                                 lidar_grid=pts_inv_b, v_i_feat=v_i_feat_b)

            # split
            st = 0
            for b in range(batch_size):
                n_ne = (encoded_voxel.indices[:, 0] == b).sum()
                for i in range(6):
                    mask = mask_n[b][i]
                    num_ne = mask.nonzero().shape[0]
                    # for now fuse by sum
                    encoded_voxel.features[st:st+n_ne][mask] += enh_feat[b*6+i][:num_ne]
                st += n_ne

            if 'gt_boxes_noaug' in example:
                if self.seg_cfg is not None:
                    batch_dict['auxseg_loss'] = auxloss_n

            return encoded_voxel
        else:
            return encoded_voxel

