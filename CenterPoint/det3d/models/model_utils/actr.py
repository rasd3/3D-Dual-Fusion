# ------------------------------------------------------------------------
# Modified Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""
ACTR module
"""
import argparse
import copy
import math

import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torch import nn

from .position_encoding import (
    PositionEmbeddingSineSparse,
    PositionEmbeddingSine,
    PositionEmbeddingSineSparseDepth,
    PositionEmbeddingLearnedDepth,
)
from .actr_transformer import build_deformable_transformer
from .actr_utils import (
    accuracy,
    get_args_parser,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
    NestedTensor,
)

IDX = 0


class ACTR(nn.Module):
    """This is the Deformable ACTR module that performs cross projection"""

    def __init__(
        self,
        transformer,
        num_channels,
        num_feature_levels,
        max_num_ne_voxel,
        p_num_channels=None,
        pos_encode_method="image_coor",
        feature_modal='lidar',
    ):
        """Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            num_channels: [List] number of feature channels to bring from Depth Network Layer
            num_feature_levels: [int] number of feature level
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        num_backbone_outs = len(num_channels)
        self.num_backbone_outs = num_backbone_outs
        assert num_backbone_outs == num_feature_levels
        if num_feature_levels > 1:
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            ])

        prior_prob = 0.01
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        # add
        if feature_modal in ['image', 'hybrid']:
            self.i_input_proj = nn.Sequential(
                nn.Conv1d(num_channels[0], hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )
            nn.init.xavier_uniform_(self.i_input_proj[0].weight, gain=1)
            nn.init.constant_(self.i_input_proj[0].bias, 0)
        self.feature_modal = feature_modal
        self.max_num_ne_voxel = max_num_ne_voxel
        self.pos_encode_method = pos_encode_method
        assert self.pos_encode_method in ["image_coor", "depth", "depth_learn"]
        if self.pos_encode_method == "image_coor":
            self.q_position_embedding = PositionEmbeddingSineSparse(
                num_pos_feats=self.transformer.q_model // 2, normalize=True)
        elif self.pos_encode_method == "depth":
            self.q_position_embedding = PositionEmbeddingSineSparseDepth(
                num_pos_feats=self.transformer.q_model, normalize=True)
        elif self.pos_encode_method == "depth_learn":
            self.q_position_embedding = PositionEmbeddingLearnedDepth(
                num_pos_feats=self.transformer.q_model)

        self.v_position_embedding = PositionEmbeddingSine(
            num_pos_feats=hidden_dim // 2, normalize=True)

    def scatter_non_empty_voxel(self,
                                v_feat,
                                q_enh_feats,
                                q_idxs,
                                in_zeros=False):
        if in_zeros:
            s_feat = torch.zeros_like(v_feat)
        else:
            s_feat = v_feat

        for idx, (q_feat, q_idx) in enumerate(zip(q_enh_feats, q_idxs)):
            q_num = q_idx.shape[0]
            q_feat_t = q_feat.transpose(1, 0)
            s_feat[idx][:, q_idx[:, 0], q_idx[:, 1],
                        q_idx[:, 2]] = q_feat_t[:, :q_num]
        return s_feat

    def forward(
        self,
        v_feat,
        grid,
        i_feats,
        v_i_feat=None,
        lidar_grid=None,
    ):
        """Parameters:
            v_feat: 3d coord sparse voxel features (B, C, X, Y, Z)
            grid: image coordinates of each v_features (B, X, Y, Z, 3)
            i_feats: image features (consist of multi-level)
            in_zeros: whether scatter to empty voxel or not

        It returns a dict with the following elements:
           - "srcs_enh": enhanced feature from camera coordinates
        """
        # get query feature & ref points
        q_feat_flattens = v_feat
        q_ref_coors = grid
        q_i_feat_flattens = None
        if self.feature_modal in ['image', 'hybrid']:
            assert v_i_feat is not None
            q_i_feat_flattens = self.i_input_proj(v_i_feat.transpose(1, 2))
            q_i_feat_flattens = q_i_feat_flattens.transpose(1, 2)
            if self.feature_modal == 'image':
                q_feat_flattens = q_i_feat_flattens

        if self.pos_encode_method == "image_coor":
            q_pos = self.q_position_embedding(q_ref_coors).transpose(1, 2)
        elif "depth" in self.pos_encode_method:
            q_depths = lidar_grid[..., 0].clone()
            q_pos = self.q_position_embedding(q_depths).transpose(1, 2)

        # get image feature with reduced channel
        pos = []
        srcs = []
        masks = []
        for l, src in enumerate(i_feats):
            s_proj = self.input_proj[l](src)
            mask = torch.zeros(
                (s_proj.shape[0], s_proj.shape[2], s_proj.shape[3]),
                dtype=torch.bool,
                device=src.device,
            )
            pos_l = self.v_position_embedding(NestedTensor(s_proj, mask)).to(
                s_proj.dtype)
            pos.append(pos_l)
            srcs.append(s_proj)
            masks.append(mask)

        q_enh_feats = self.transformer(srcs, masks, pos, q_feat_flattens,
                                       q_pos, q_ref_coors, q_lidar_grid=lidar_grid,
                                       q_i_feat_flatten=q_i_feat_flattens
                                       )

        return q_enh_feats


class IACTR(nn.Module):
    """This is the Deformable IACTR module that performs cross projection"""

    def __init__(
        self,
        transformer,
        num_channels,
        p_num_channels,
        num_feature_levels,
        max_num_ne_voxel,
        pos_encode_method="image_coor",
    ):
        """Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            num_channels: [List] number of feature channels to bring from Depth Network Layer
            num_feature_levels: [int] number of feature level
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        num_backbone_outs = len(num_channels)
        self.num_backbone_outs = num_backbone_outs
        assert num_backbone_outs == num_feature_levels
        if num_feature_levels > 1:
            i_input_proj_list = []
            p_input_proj_list = []
            for _ in range(num_backbone_outs):
                i_in_channels = num_channels[_]
                p_in_channels = p_num_channels[_]
                i_input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(i_in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                p_input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(p_in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
            self.i_input_proj = nn.ModuleList(i_input_proj_list)
            self.p_input_proj = nn.ModuleList(p_input_proj_list)
        else:
            self.i_input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            ])
            self.p_input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            ])

        prior_prob = 0.01
        for i_proj, p_proj in zip(self.i_input_proj, self.p_input_proj):
            nn.init.xavier_uniform_(i_proj[0].weight, gain=1)
            nn.init.xavier_uniform_(p_proj[0].weight, gain=1)
            nn.init.constant_(i_proj[0].bias, 0)
            nn.init.constant_(p_proj[0].bias, 0)
        # add
        self.max_num_ne_voxel = max_num_ne_voxel
        self.pos_encode_method = pos_encode_method

        self.i_position_embedding = PositionEmbeddingSine(
            num_pos_feats=hidden_dim // 2, normalize=True)
        self.p_position_embedding = PositionEmbeddingSine(
            num_pos_feats=hidden_dim // 2, normalize=True)

    def scatter_non_empty_voxel(
        self,
        v_feat,
        q_enh_feats,
        q_idxs,
        in_zeros=False,
    ):
        if in_zeros:
            s_feat = torch.zeros_like(v_feat)
        else:
            s_feat = v_feat

        for idx, (q_feat, q_idx) in enumerate(zip(q_enh_feats, q_idxs)):
            q_num = q_idx.shape[0]
            q_feat_t = q_feat.transpose(1, 0)
            s_feat[idx][:, q_idx[:, 0], q_idx[:, 1],
                        q_idx[:, 2]] = q_feat_t[:, :q_num]
        return s_feat

    def forward(
        self,
        i_feats,
        p_feats,
        ret_pts_img=False,
    ):
        """Parameters:
            v_feat: 3d coord sparse voxel features (B, C, X, Y, Z)
            grid: image coordinates of each v_features (B, X, Y, Z, 3)
            i_feats: image features (consist of multi-level)
            in_zeros: whether scatter to empty voxel or not

        It returns a dict with the following elements:
           - "srcs_enh": enhanced feature from camera coordinates
        """

        # get image feature with reduced channel
        i_pos, p_pos = [], []
        i_srcs, p_srcs = [], []
        masks = []
        for l, (i_src, p_src) in enumerate(zip(i_feats, p_feats)):
            i_proj = self.i_input_proj[l](i_src)
            p_proj = self.p_input_proj[l](p_src)
            mask = torch.zeros(
                (i_proj.shape[0], i_proj.shape[2], i_proj.shape[3]),
                dtype=torch.bool,
                device=i_src.device,
            )
            pos_i = self.i_position_embedding(NestedTensor(i_proj, mask)).to(
                i_proj.dtype)
            pos_p = self.p_position_embedding(NestedTensor(p_proj, mask)).to(
                p_proj.dtype)
            i_pos.append(pos_i)
            p_pos.append(pos_p)
            i_srcs.append(i_proj)
            p_srcs.append(p_proj)
            masks.append(mask)

        if ret_pts_img:
            return p_srcs

        q_enh_feats = self.transformer(p_srcs, masks, p_pos, i_srcs, i_pos)

        return q_enh_feats


class IACTRv2(IACTR):
    """This is the Deformable IACTR module that performs cross projection"""

    def visualize(self, i_feats, i_enh_feats, p_feats):
        global IDX
        for s in range(len(i_feats)):
            for b in range(batch_size):
                i_feat = i_enh_feats[s][b].max(0)[0].detach().cpu().numpy()
                i_feat = (i_feat - i_feat.min()) / (i_feat.max() -
                                                    i_feat.min()) * 255.
                i_feat = i_feat.astype(np.uint8)
                cv2.imwrite('./vis/ifatv2_%d_%d.png' % (IDX + b, s), i_feat)
                i_feat = i_feats[s][b].max(0)[0].detach().cpu().numpy()
                i_feat = (i_feat - i_feat.min()) / (i_feat.max() -
                                                    i_feat.min()) * 255.
                i_feat = i_feat.astype(np.uint8)
                cv2.imwrite('./vis/ifeat_%d_%d.png' % (IDX + b, s), i_feat)
                i_feat = p_feats[s][b].max(0)[0].detach().cpu().numpy()
                i_feat = (i_feat - i_feat.min()) / (i_feat.max() -
                                                    i_feat.min()) * 255.
                i_feat = i_feat.astype(np.uint8)
                cv2.imwrite('./vis/ipfeat_%d_%d.png' % (IDX + b, s), i_feat)
        IDX += batch_size

    def forward(
        self,
        i_feats,
        p_feats,
        ret_pts_img=False,
    ):
        """Parameters:
            v_feat: 3d coord sparse voxel features (B, C, X, Y, Z)
            grid: image coordinates of each v_features (B, X, Y, Z, 3)
            i_feats: image features (consist of multi-level)
            in_zeros: whether scatter to empty voxel or not

        It returns a dict with the following elements:
           - "srcs_enh": enhanced feature from camera coordinates
        """
        batch_size = i_feats[0].shape[0]

        # get image feature with reduced channel
        p_srcs, p_pos = [], []
        i_srcs, i_pos = [[] for _ in range(batch_size)
                         ], [[] for _ in range(batch_size)]
        i_nz_ns, i_nzs = [[] for _ in range(batch_size)
                          ], [[] for _ in range(batch_size)]
        max_ne_voxel = []
        masks = []
        for l, (i_src, p_src) in enumerate(zip(i_feats, p_feats)):
            i_proj = self.i_input_proj[l](i_src)
            p_proj = self.p_input_proj[l](p_src)
            mask = torch.zeros(
                (i_proj.shape[0], i_proj.shape[2], i_proj.shape[3]),
                dtype=torch.bool,
                device=i_src.device,
            )
            pos_i = self.i_position_embedding(NestedTensor(i_proj, mask)).to(
                i_proj.dtype)
            pos_p = self.p_position_embedding(NestedTensor(p_proj, mask)).to(
                p_proj.dtype)

            max_v = 0
            for b in range(batch_size):
                i_nz = torch.nonzero(p_src[b].max(0)[0])
                i_nz_n = i_nz.to(torch.float) / torch.tensor(
                    p_src[0].shape[1:]).cuda()
                i_proj_nz = i_proj[b, :, i_nz[:, 0], i_nz[:, 1]]
                pos_i_nz = pos_i[b, :, i_nz[:, 0], i_nz[:, 1]]
                max_v = max(max_v, i_nz.shape[0])
                i_nzs[b].append(i_nz)
                i_nz_ns[b].append(i_nz_n)
                i_srcs[b].append(i_proj_nz)
                i_pos[b].append(pos_i_nz)
            max_ne_voxel.append(max_v)

            p_pos.append(pos_p)
            p_srcs.append(p_proj + i_proj)
            masks.append(mask)

        i_srcs_t_l, i_nz_ns_t_l, i_pos_t_l = [], [], []
        for s in range(len(i_feats)):
            i_nz_ns_t = torch.zeros((batch_size, max_ne_voxel[s], 2),
                                    device=i_feats[0].device)
            i_srcs_t = torch.zeros(
                (batch_size, i_srcs[0][0].shape[0], max_ne_voxel[s]),
                device=i_feats[0].device)
            i_pos_t = torch.zeros(
                (batch_size, i_pos[0][0].shape[0], max_ne_voxel[s]),
                device=i_feats[0].device)
            for b in range(batch_size):
                n_point = i_nz_ns[b][s].shape[0]
                i_nz_ns_t[b, :n_point] = i_nz_ns[b][s]
                i_srcs_t[b, :, :n_point] = i_srcs[b][s]
                i_pos_t[b, :, :n_point] = i_pos[b][s]
            i_nz_ns_t_l.append(i_nz_ns_t)
            i_srcs_t_l.append(i_srcs_t)
            i_pos_t_l.append(i_pos_t)

        i_nz_ns_t_l = torch.cat(i_nz_ns_t_l, dim=1)
        q_enh_feats = self.transformer(
            p_srcs,
            masks,
            p_pos,
            i_srcs_t_l,
            i_pos_t_l,
            q_ref_coors=i_nz_ns_t_l)

        i_enh_feats = [
            torch.zeros_like(i_feats[s]) for s in range(len(i_feats))
        ]
        ne_cum = torch.tensor([0] + max_ne_voxel).cumsum(0)
        for b in range(batch_size):
            for s in range(len(i_feats)):
                coor = i_nzs[b][s]
                q_enh_feat = q_enh_feats[b, ne_cum[s]:ne_cum[s] +
                                         i_nzs[b][s].shape[0]]
                i_enh_feats[s][b][:, coor[:, 0],
                                  coor[:, 1]] = q_enh_feat.permute(1, 0)

        if False:
            self.visualize(i_feats, i_enh_feats, p_feats)

        return i_enh_feats


class IACTRv3(IACTR):
    """This is the Deformable IACTR module that performs cross projection"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pos_encode_method = kwargs['pos_encode_method']
        if 'depth' in self.pos_encode_method:
            self.i_position_embedding = PositionEmbeddingSineSparseDepth(
                num_pos_feats=self.transformer.q_model, normalize=True)
        if self.pos_encode_method == 'depth_v2':
            self.i_position_embedding_ori = PositionEmbeddingSine(
                num_pos_feats=self.transformer.d_model // 2, normalize=True)

    def visualize(self, i_feats, i_enh_feats, p_feats):
        global IDX
        for s in range(len(i_feats)):
            for b in range(batch_size):
                i_feat = i_enh_feats[s][b].max(0)[0].detach().cpu().numpy()
                i_feat = (i_feat - i_feat.min()) / (i_feat.max() -
                                                    i_feat.min()) * 255.
                i_feat = i_feat.astype(np.uint8)
                cv2.imwrite('./vis/ifatv2_%d_%d.png' % (IDX + b, s), i_feat)
                i_feat = i_feats[s][b].max(0)[0].detach().cpu().numpy()
                i_feat = (i_feat - i_feat.min()) / (i_feat.max() -
                                                    i_feat.min()) * 255.
                i_feat = i_feat.astype(np.uint8)
                cv2.imwrite('./vis/ifeat_%d_%d.png' % (IDX + b, s), i_feat)
                i_feat = p_feats[s][b].max(0)[0].detach().cpu().numpy()
                i_feat = (i_feat - i_feat.min()) / (i_feat.max() -
                                                    i_feat.min()) * 255.
                i_feat = i_feat.astype(np.uint8)
                cv2.imwrite('./vis/ipfeat_%d_%d.png' % (IDX + b, s), i_feat)
        IDX += batch_size

    def forward(
        self,
        i_feats,
        p_feats,
        p_depths,
        ret_pts_img=False,
    ):
        """Parameters:
            v_feat: 3d coord sparse voxel features (B, C, X, Y, Z)
            grid: image coordinates of each v_features (B, X, Y, Z, 3)
            i_feats: image features (consist of multi-level)
            in_zeros: whether scatter to empty voxel or not

        It returns a dict with the following elements:
           - "srcs_enh": enhanced feature from camera coordinates
        """
        batch_size = i_feats[0].shape[0]

        # get image feature with reduced channel
        p_srcs, p_pos = [], []
        i_srcs, i_pos = [[] for _ in range(batch_size)
                         ], [[] for _ in range(batch_size)]
        i_nz_ns, i_nzs = [[] for _ in range(batch_size)
                          ], [[] for _ in range(batch_size)]
        max_ne_voxel = []
        masks = []
        for l, (i_src, p_src,
                p_depth) in enumerate(zip(i_feats, p_feats, p_depths)):
            i_proj = self.i_input_proj[l](i_src)
            p_proj = self.p_input_proj[l](p_src)
            mask = torch.zeros(
                (i_proj.shape[0], i_proj.shape[2], i_proj.shape[3]),
                dtype=torch.bool,
                device=i_src.device,
            )
            if self.pos_encode_method == 'depth_v2':
                pos_i = self.i_position_embedding_ori(
                    NestedTensor(i_proj, mask)).to(i_proj.dtype)
            pos_p = self.p_position_embedding(NestedTensor(p_proj, mask)).to(
                p_proj.dtype)

            max_v = 0
            for b in range(batch_size):
                i_nz = torch.nonzero(p_src[b].max(0)[0])
                max_v = max(max_v, i_nz.shape[0])
                i_nz_n = i_nz.to(torch.float) / torch.tensor(
                    p_src[0].shape[1:]).cuda()
                i_proj_nz = i_proj[b, :, i_nz[:, 0], i_nz[:, 1]]

                # position encoding
                p_depth_nz = p_depth[b, :, i_nz[:, 0], i_nz[:, 1]]
                pos_i_nz = self.i_position_embedding(
                    p_depth_nz[0].unsqueeze(0))[0]
                if self.pos_encode_method == 'depth_v2':
                    pos_i_nz_img_coor = pos_i[b, :, i_nz[:, 0], i_nz[:, 1]]
                    pos_i_nz += pos_i_nz_img_coor

                i_nzs[b].append(i_nz)
                i_nz_ns[b].append(i_nz_n)
                i_srcs[b].append(i_proj_nz)
                i_pos[b].append(pos_i_nz)
            max_ne_voxel.append(max_v)

            p_pos.append(pos_p)
            p_srcs.append(p_proj + i_proj)
            masks.append(mask)

        i_srcs_t_l, i_nz_ns_t_l, i_pos_t_l = [], [], []
        for s in range(len(i_feats)):
            i_nz_ns_t = torch.zeros((batch_size, max_ne_voxel[s], 2),
                                    device=i_feats[0].device)
            i_srcs_t = torch.zeros(
                (batch_size, i_srcs[0][0].shape[0], max_ne_voxel[s]),
                device=i_feats[0].device)
            i_pos_t = torch.zeros(
                (batch_size, i_pos[0][0].shape[0], max_ne_voxel[s]),
                device=i_feats[0].device)
            for b in range(batch_size):
                n_point = i_nz_ns[b][s].shape[0]
                i_nz_ns_t[b, :n_point] = i_nz_ns[b][s]
                i_srcs_t[b, :, :n_point] = i_srcs[b][s]
                i_pos_t[b, :, :n_point] = i_pos[b][s]
            i_nz_ns_t_l.append(i_nz_ns_t)
            i_srcs_t_l.append(i_srcs_t)
            i_pos_t_l.append(i_pos_t)

        i_nz_ns_t_l = torch.cat(i_nz_ns_t_l, dim=1)
        q_enh_feats = self.transformer(
            p_srcs,
            masks,
            p_pos,
            i_srcs_t_l,
            i_pos_t_l,
            q_ref_coors=i_nz_ns_t_l)

        i_enh_feats = [
            torch.zeros_like(i_feats[s]) for s in range(len(i_feats))
        ]
        ne_cum = torch.tensor([0] + max_ne_voxel).cumsum(0)
        for b in range(batch_size):
            for s in range(len(i_feats)):
                coor = i_nzs[b][s]
                q_enh_feat = q_enh_feats[b, ne_cum[s]:ne_cum[s] +
                                         i_nzs[b][s].shape[0]]
                i_enh_feats[s][b][:, coor[:, 0],
                                  coor[:, 1]] = q_enh_feat.permute(1, 0)

        if False:
            self.visualize(i_feats, i_enh_feats, p_feats)

        return i_enh_feats


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(model_cfg, model_name='ACTR', lt_cfg=None):
    parser = argparse.ArgumentParser(
        "Deformable DETR training and evaluation script",
        parents=[get_args_parser()])
    args = parser.parse_args([])
    device = torch.device(args.device)
    model_dict = {
        'ACTR': ACTR,
        'ACTRv2': ACTR,
        'IACTR': IACTR,
        'IACTRv2': IACTRv2,
        'IACTRv3': IACTRv3
    }

    # from yaml
    num_channels = model_cfg.num_channels
    args.query_num_feat = model_cfg.query_num_feat
    args.hidden_dim = model_cfg.query_num_feat
    args.enc_layers = model_cfg.num_enc_layers
    args.pos_encode_method = model_cfg.pos_encode_method
    args.max_num_ne_voxel = model_cfg.max_num_ne_voxel
    args.num_feature_levels = len(model_cfg.num_channels)
    args.feature_modal = model_cfg.get('feature_modal', 'lidar')
    args.hybrid_cfg = model_cfg.get('hybrid_cfg', None)

    model_class = model_dict[model_name]
    transformer = build_deformable_transformer(args, model_name=model_name, lt_cfg=lt_cfg)

    model = model_class(
        transformer,
        num_feature_levels=args.num_feature_levels,
        p_num_channels=model_cfg.get('p_num_channels', None),
        num_channels=num_channels,
        max_num_ne_voxel=args.max_num_ne_voxel,
        pos_encode_method=args.pos_encode_method,
        feature_modal=args.feature_modal
    )

    return model
