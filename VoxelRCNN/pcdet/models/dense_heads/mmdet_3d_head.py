# 2D detection head based on mmdetection.


import numpy as np
import torch
import torch.nn as nn

from mmdet3d.models.builder import build_head
from mmdet3d.core import bbox3d2result
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
import copy
from mmdet3d.core.bbox import Box3DMode, CameraInstance3DBoxes, get_box_type
import torch.nn.functional as F

class MMDet3DHead(nn.Module):
    def __init__(self, model_cfg, feat_lvl):
        super(MMDet3DHead, self).__init__()
        model_cfg.cfg.strides = model_cfg.cfg.strides[feat_lvl[0]:feat_lvl[0]+len(feat_lvl)]
        model_cfg.cfg.regress_ranges = model_cfg.cfg.regress_ranges[feat_lvl[0]:feat_lvl[0]+len(feat_lvl)]
        self.bbox_head = build_head(model_cfg.cfg)
        self.bbox_head.init_weights()

        # if getattr(model_cfg, 'load_from') is not None:
        #     from mmcv.runner.checkpoint import load_state_dict
        #     state_dict = torch.load(model_cfg.load_from, map_location='cpu')['state_dict']
        #     print('loading mmdet head from ', model_cfg.load_from)
        #     load_state_dict(self.bbox_head, {k[10:]: v for k, v in state_dict.items() if k.startswith("bbox_head.")}, strict=False, logger=None)
        
        # pcdet = Ped : 2 Cyc : 3 Car : 1
        # mmdet = Ped : 0 Cyc : 1 Car : 2


    def get_loss(self, data_dict, tb_dict):

        img_metas = [{
            "image": data_dict['images'][i],  # for debug
            "filename": "data/kitti/training/image_2/" + data_dict['frame_id'][i] + ".png",
            "img_shape": list(map(int,data_dict['image_shape'][0].cpu())),
            "ori_shape": list(data_dict['images'][i].shape[1:3]) + [3],
            "pad_shape": list(map(int,data_dict['pad_shape'][0].cpu())) if 'pad_shape' in data_dict else None,
            'img_norm_cfg': data_dict['img_norm_cfg'][i] if 'img_norm_cfg' in data_dict else None,
            'scale_factor': data_dict['scale_factor'][i].cpu().numpy() if 'scale_factor' in data_dict else None,
            'cam2img': data_dict['trans_cam_to_img'][i].cpu().tolist()+[[0,0,0,1]],
            'transformation_3d_flow': data_dict['transformation_3d_flow'][0] if 'transformation_3d_flow' in data_dict else [],
            'box_type_3d': get_box_type('Camera')[0],
            'box_mode_3d': get_box_type('Camera')[1]
            }
            for i in range(len(data_dict['images']))]
        
        gt_boxes_2d = data_dict['gt_boxes2d_no3daug'][:,:,:-1]
        gt_labels_2d = data_dict['gt_boxes2d_no3daug'][:,:,-1]
        data_dict['mmdet_2d_gt_labels'] = torch.unbind(gt_labels_2d.long() - 1)
        #data_dict['mmdet_2d_gt_labels'] = self.pcdet2mmdet_label(gt_labels_2d)
        data_dict['gt_boxes2d_no3daug'] = torch.unbind(gt_boxes_2d)

        gt_boxes = data_dict['gt_boxes_no3daug'][:,:,:-1]
        gt_labels = data_dict['gt_boxes_no3daug'][:,:,-1]
        device = gt_boxes.device
        data_dict['mmdet_gt_labels'] = torch.unbind(gt_labels.long() - 1)
        #data_dict['mmdet_gt_labels'] = self.pcdet2mmdet_label(gt_labels)

        center2d_list = []
        gt_boxes_cam_list = []
        depth_list = []

        for batch_idx in range(len(data_dict['gt_boxes_no3daug'])):
            gt_boxes_cam = box_utils.boxes3d_lidar_to_kitti_camera(gt_boxes[batch_idx].cpu(), data_dict['calib'][batch_idx])
            gt_boxes_cam = torch.tensor(gt_boxes_cam)
            center3d = gt_boxes_cam[:,:3]
            center3d[:,1] -= gt_boxes_cam[:,4] / 2
            cam2img = torch.vstack((data_dict['trans_cam_to_img'][batch_idx].cpu(),torch.tensor([0,0,0,1]))).numpy()
            center2d = self.points_cam2img(center3d, cam2img, with_depth=True).to(device=device)
            center2d_list.append(center2d[:,:2])
            depth_list.append(center2d[:,2])
            gt_boxes_cam = CameraInstance3DBoxes(
                                gt_boxes_cam,
                                box_dim=gt_boxes_cam.shape[-1],
                                origin=(0.5, 0.5, 0.5))
            gt_boxes_cam_list.append(gt_boxes_cam)

        data_dict['depths'] = depth_list
        data_dict['centers2d'] = center2d_list
        data_dict['gt_boxes_cam'] = gt_boxes_cam_list
        
        # # centers 2d 위치 확인, batch 1에서만 가능
        # import cv2
        # img22 = data_dict['images'][0].cpu().permute((1,2,0)).numpy()
        # img2 = img22.copy()
        # for i in range(len(center2d)):
        #     cv2.circle(img2,(int(center2d[i,:2][0]),int(center2d[i,:2][1])),5,(255,255,255),-1)
        # cv2.imwrite("1.jpg",img2)

        # # 3d box bottom center 확인, batch 1에서만 가능
        # import cv2
        # centre3d = gt_boxes_cam.tensor.cpu()[:,:3]
        # centre2d = self.points_cam2img(centre3d, cam2img, with_depth=True)
        # img22 = data_dict['images'][0].cpu().permute((1,2,0)).numpy()
        # img2 = img22.copy()
        # for i in range(len(centre2d)):
        #     cv2.circle(img2,(int(centre2d[i,:2][0]),int(centre2d[i,:2][1])),5,(255,255,255),-1)
        # cv2.imwrite("2.jpg",img2)

        # # 2d box 확인, batch 1에서만 가능
        # import cv2
        # img22 = data_dict['images'][0].cpu().permute((1,2,0)).numpy()
        # img2 = img22.copy()
        # box2d = data_dict['gt_boxes2d_no3daug']
        # for i in range(len(box2d[0])):
        #     cv2.rectangle(img2,(int(box2d[0][i,0]),int(box2d[0][i,1])),(int(box2d[0][i,2]),int(box2d[0][i,3])),(255,255,255),2)
        # cv2.imwrite("3.jpg",img2)



        if len(data_dict['gt_boxes2d_no3daug'][0]) != len(data_dict['gt_boxes_cam'][0]):
            import pdb;pdb.set_trace()

        losses = self.bbox_head.forward_train(data_dict['img_feats'], \
        img_metas, data_dict['gt_boxes2d_no3daug'], data_dict['mmdet_2d_gt_labels'],\
             data_dict['gt_boxes_cam'], data_dict['mmdet_gt_labels'],\
                  data_dict['centers2d'], data_dict['depths'])
        

        for k, v in losses.items():
            if not isinstance(v, (list, tuple)) and len(v.shape) == 0:
                _sum_loss = v
            else:
                _sum_loss = sum(_loss for _loss in v)
            assert len(_sum_loss.shape) == 0
            # if k != 'loss_bbox':
            #     assert len(_sum_loss.shape) == 0
            # else:
            #     assert len(_sum_loss.shape) in [0, 1]
            #     if len(_sum_loss.shape) == 1:
            #         assert _sum_loss.shape[0] < 10
            # if len(_sum_loss.shape) == 1:
            #     for i in range(_sum_loss.shape[0]):
            #         tb_dict['rpn2d_' + k + '_' + str(i)] = _sum_loss[i].item()
            losses[k] = _sum_loss.sum()
            tb_dict['rpn3d_' + k] = losses[k].item()
        loss_sum = sum([v for _, v in losses.items()])
        return loss_sum, tb_dict



    def points_cam2img(self, points_3d, proj_mat, with_depth=False):
        """Project points in camera coordinates to image coordinates.

        Args:
            points_3d (torch.Tensor | np.ndarray): Points in shape (N, 3)
            proj_mat (torch.Tensor | np.ndarray):
                Transformation matrix between coordinates.
            with_depth (bool, optional): Whether to keep depth in the output.
                Defaults to False.

        Returns:
            (torch.Tensor | np.ndarray): Points in image coordinates,
                with shape [N, 2] if `with_depth=False`, else [N, 3].
        """
        points_shape = list(points_3d.shape)
        points_shape[-1] = 1

        assert len(proj_mat.shape) == 2, 'The dimension of the projection'\
            f' matrix should be 2 instead of {len(proj_mat.shape)}.'
        d1, d2 = proj_mat.shape[:2]
        assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (
            d1 == 4 and d2 == 4), 'The shape of the projection matrix'\
            f' ({d1}*{d2}) is not supported.'
        if d1 == 3:
            proj_mat_expanded = torch.eye(
                4, device=proj_mat.device, dtype=proj_mat.dtype)
            proj_mat_expanded[:d1, :d2] = proj_mat
            proj_mat = proj_mat_expanded

        # previous implementation use new_zeros, new_one yields better results
        points_4 = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)
        
        points_4 = points_4.cpu()

        point_2d = points_4 @ proj_mat.T


        point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

        if with_depth:
            point_2d_res = torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)

        return point_2d_res

    def pcdet2mmdet_label(self, pcdet_label):

        mmdet_gt_labels = torch.full(pcdet_label.shape,3).to(device=pcdet_label.device)
        mmdet_gt_labels[(pcdet_label==1).nonzero(as_tuple=True)[0],(pcdet_label==1).nonzero(as_tuple=True)[1]] = 2
        mmdet_gt_labels[(pcdet_label==2).nonzero(as_tuple=True)[0],(pcdet_label==2).nonzero(as_tuple=True)[1]] = 0
        mmdet_gt_labels[(pcdet_label==3).nonzero(as_tuple=True)[0],(pcdet_label==3).nonzero(as_tuple=True)[1]] = 1
        mmdet_gt_labels = torch.unbind(mmdet_gt_labels)

        return mmdet_gt_labels

    def forward(self, data_dict):
        if self.training:
            return data_dict
        else:
            pass
            return data_dict
