import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from ..vfe.image_vfe_modules.ffn.ddn_loss.balancer import Balancer
from pcdet.utils import transform_utils, loss_utils
from pcdet.utils import common_utils
import numpy as np
from .focalloss_segmentation import FocalLoss

# from focal_sparse_conv.utils import FocalLoss
# try:
#     from kornia.losses.focal import FocalLoss
# except:
#     pass
# print('Warning: kornia is not installed. This package is only required by CaDDN')


class AuxImgSegmentLoss(nn.Module):
    def __init__(self, weight, alpha, gamma, fg_weight, bg_weight, downsample_factor):
        """
        Initializes DDNLoss module
        Args:
            weight: float, Loss function weight
            alpha: float, Alpha value for Focal Loss
            gamma: float, Gamma value for Focal Loss
            disc_cfg: dict, Depth discretiziation configuration
            fg_weight: float, Foreground loss weight
            bg_weight: float, Background loss weight
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.device = torch.cuda.current_device()
        # self.balancer = Balancer(downsample_factor=downsample_factor,
        #                          fg_weight=fg_weight,
        #                          bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        # self.loss_func = FocalLoss()
        self.loss_func = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")
        self.weight = weight
        use_conv_for_no_stride = None
        upsample_strides = [0.5, 1, 2, 4]
        deblocks = []
        in_channels = 256
        out_channel = 256
        self.num_classes = 1  # foreground : 1 background : 0
        self.downsample_factor = downsample_factor
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        # self.balancer = Balancer(downsample_factor=downsample_factor,
        #                                 fg_weight=fg_weight,
        #                                 bg_weight=bg_weight)

        for i in range(len(upsample_strides)):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = torch.nn.ConvTranspose2d(
                    in_channels,
                    out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i],
                )

            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = torch.nn.Upsample(scale_factor=upsample_strides[i])
                # upsample_layer = torch.nn.Upsample(input = img_aux_feats, scale_factor=upsample_strides[i])
            deblock = nn.Sequential(
                upsample_layer, torch.nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True)
            )
            deblocks.append(deblock)
        self.aux_img_cls_agg = nn.ModuleList(deblocks)
        self.aux_img_cls = nn.Conv2d(out_channel, self.num_classes, 1)

    def pred_fg_img(self, batch_dict):

        cat_list = []
        aux_img_losses_cls = torch.tensor(0.0).cuda()
        batch_size = batch_dict["batch_size"]
        img_aux_feats = batch_dict["img_dict"]

        for i in range(len(img_aux_feats)):
            if i == 0:
                img_aux_feats[0] = img_aux_feats["layer1_feat2d"]
            cat_list.append(self.aux_img_cls_agg[i](img_aux_feats[i]))
        # cat_list.append(self.aux_img_cls_agg(img_aux_feats))

        cat_feat = torch.cat(cat_list, dim=1)

        pred_fg_img = self.aux_img_cls(cat_feat)
        target_size = self.aux_img_cls(cat_feat).shape
        # gt_cls_img = nn.functional.interpolate(
        #     img_mask.unsqueeze(1), size=pred_fg_img.shape[2:]).unsqueeze(dim=2)

        pred_fg_img_list = pred_fg_img.permute(0, 2, 3, 1).clone()
        # pred_fg_img = pred_fg_img.permute(0, 2, 3, 1).reshape(
        #     batch_size, -1, self.num_classes).squeeze(2).unsqueeze(1)

        # gt_cls_img = gt_cls_img.squeeze().reshape(batch_size, -1).squeeze(1).float()
        # for b in range(batch_size):
        #     aux_img_loss_cls = self.aux_img_loss_cls(
        #         pred_cls_img[b], gt_cls_img[b].to(torch.long))
        #     aux_img_losses_cls += aux_img_loss_cls
        batch_dict["pred_aux_img_seg"] = pred_fg_img
        return pred_fg_img, target_size

    def forward(self, batch_dict, tb_dict):
        """
        Gets DDN loss
        Args:
            depth_logits: (B, D+1, H, W), Predicted depth logits
            depth_maps: (B, H, W), Depth map [m]
            gt_boxes2d: torch.Tensor (B, N, 4), 2D box labels for foreground/background balancing
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        gt_boxes2d = batch_dict["gt_boxes2d"]
        images = batch_dict["images"]
        images_rei = torch.zeros(
            images.shape[0], 1, images.shape[2], images.shape[3], device="cuda"
        )

        fg_pred, target_size = self.pred_fg_img(batch_dict)
        target_size = torch.zeros(target_size[0], target_size[2], target_size[3])
        # Compute loss
        # loss = self.loss_func(depth_logits, depth_target)

        fg_mask = loss_utils.compute_fg_mask(
            gt_boxes2d=gt_boxes2d,
            shape=images_rei.shape,
            downsample_factor=self.downsample_factor,
            device=images_rei.device,
        )
        fg_mask = nn.functional.interpolate(
            fg_mask.to(torch.float), size=target_size.shape[1:]
        ).squeeze(dim=1)
        # fg_mask = fg_mask.permute(0, 2, 3, 1).reshape(batch_dict['batch_size'], -1, self.num_classes).squeeze(2).unsqueeze(1).to(torch.int64)

        # Compute loss

        # Bin depth map to create target
        # depth_target = transform_utils.bin_depths(depth_maps, **self.disc_cfg, target=True)

        # Compute foreground/background balancing
        # loss, tb_dict = self.balancer(loss=loss.unsqueeze(dim=1), fg_mask=fg_mask.to(torch.bool))
        _fg_mask = fg_mask.to(torch.long)
        fg_mask = fg_mask.to(torch.bool)
        bg_mask = ~fg_mask
        _bg_mask = bg_mask.to(torch.long)
        # Compute balancing weights
        weights = self.fg_weight * fg_mask + self.bg_weight * bg_mask
        num_pixels = fg_mask.sum() + bg_mask.sum()

        # Compute losses
        loss = self.loss_func(fg_pred, fg_mask.squeeze(dim=1).to(torch.int64))
        loss *= weights
        fg_loss = torch.mul(loss, _fg_mask).sum() / num_pixels
        bg_loss = torch.mul(loss, _bg_mask).sum() / num_pixels

        # Get total loss
        loss = fg_loss + bg_loss
        tb_dict.update(
            {
                "balancer_loss": loss.item(),
                "fg_loss": fg_loss.item(),
                "bg_loss": bg_loss.item(),
            }
        )

        # Final loss
        loss *= self.weight

        return loss, tb_dict


class AuxConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.aux_cons_loss_func = F.binary_cross_entropy

    def make_projected_img(self, batch_dict, aux_pts_dict):
        calibs = batch_dict["calib"]
        batch_size = batch_dict["batch_size"]
        inv_idx = torch.Tensor([0, 1, 2]).long().cuda()
        h, w = batch_dict["images"].shape[2:]
        nh, nw = batch_dict["pred_aux_img_seg"].shape[2:]
        ratio = round(h / batch_dict["pred_aux_img_seg"].shape[2])
        proj_img = torch.zeros([batch_size, nh, nw]).cuda()

        for b in range(batch_size):
            calib = calibs[b]
            mask = aux_pts_dict["point_coords"][:, 0] == b
            voxels_3d_batch = aux_pts_dict["point_coords"][mask][:, 1:]

            # Reverse the point cloud transformations to the original coords.
            if "noise_scale" in batch_dict:
                voxels_3d_batch[:, :3] /= batch_dict["noise_scale"][b]
            if "noise_rot" in batch_dict:
                voxels_3d_batch = common_utils.rotate_points_along_z(
                    voxels_3d_batch[:, inv_idx].unsqueeze(0),
                    -batch_dict["noise_rot"][b].unsqueeze(0),
                )[0, :, inv_idx]
            if "flip_x" in batch_dict:
                voxels_3d_batch[:, 1] *= -1 if batch_dict["flip_x"][b] else 1
            if "flip_y" in batch_dict:
                voxels_3d_batch[:, 2] *= -1 if batch_dict["flip_y"][b] else 1

            voxels_2d, _ = calib.lidar_to_img(voxels_3d_batch[:, inv_idx].cpu().numpy())
            voxels_2d_norm = voxels_2d / np.array([w, h])

            voxels_2d_int = torch.Tensor(voxels_2d).cuda().long()

            filter_idx = (
                (0 <= voxels_2d_norm[:, 1])
                * (voxels_2d_norm[:, 1] < 1)
                * (0 <= voxels_2d_norm[:, 0])
                * (voxels_2d_norm[:, 0] < 1)
            )

            pts_cls_pred = aux_pts_dict["point_cls_preds"][mask][filter_idx]
            if False:
                pts_cls_pred = pts_cls_pred.cpu().detach().numpy()
                pts_cls_pred = (
                    (pts_cls_pred - pts_cls_pred.min())
                    / (pts_cls_pred.max() - pts_cls_pred.min())
                    * 255.0
                ).astype(np.uint8)
            voxels_proj = (voxels_2d_norm[filter_idx] * np.array([nw, nh])).astype(
                np.int
            )
            proj_img[
                b, voxels_proj[:, 1], voxels_proj[:, 0]
            ] = pts_cls_pred.squeeze().sigmoid()
            #  cv2.imwrite('test.png', proj_img[b])

        return proj_img

    def forward(self, batch_dict, aux_pts_dict, tb_dict=None):
        batch_size = len(batch_dict["frame_id"])
        conf_thres = 0.2

        aux_pts_imgs = self.make_projected_img(batch_dict, aux_pts_dict)
        aux_img_imgs = batch_dict["pred_aux_img_seg"][:, 0].sigmoid()

        pts_idx = aux_pts_imgs.nonzero()
        pts_nz_list = aux_pts_imgs[pts_idx[:, 0], pts_idx[:, 1], pts_idx[:, 2]]
        img_nz_list = aux_img_imgs[pts_idx[:, 0], pts_idx[:, 1], pts_idx[:, 2]]
        mask = torch.logical_or((pts_nz_list > conf_thres), (img_nz_list > conf_thres))
        pts_nz_list = pts_nz_list[mask]
        img_nz_list = img_nz_list[mask]
        reg_weights = torch.ones_like(pts_nz_list, dtype=torch.float)
        reg_weights /= pts_nz_list.shape[0]

        consistency_loss_src_1 = self.aux_cons_loss_func(
            pts_nz_list, img_nz_list.detach(), reduction="none"
        )
        consistency_loss_src_2 = self.aux_cons_loss_func(
            img_nz_list, pts_nz_list.detach(), reduction="none"
        )
        consistency_loss = (consistency_loss_src_1 + consistency_loss_src_2) / 2
        consistency_loss = (consistency_loss * reg_weights).sum() / batch_size

        return consistency_loss, tb_dict
