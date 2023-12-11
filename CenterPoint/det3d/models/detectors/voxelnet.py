from ..registry import DETECTORS
from .. import builder
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
import torch 
from copy import deepcopy 
import time

@DETECTORS.register_module
class VoxelNet(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        self.total_time = 0
        self.iters = 0

    def extract_feat(self, data):
        input_features = self.reader(data["features"], data["num_voxels"])
        x, voxel_feature = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )
        start_time = time.time()
        x, _ = self.extract_feat(data)
        preds = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            boxes = self.bbox_head.predict(example, preds, self.test_cfg)
            end_time = time.time()
            self.total_time += (end_time - start_time)
            self.iters += 1
            #print('avg_speed', self.total_time/self.iters)
            return boxes

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x, voxel_feature = self.extract_feat(data)
        bev_feature = x 
        preds = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, voxel_feature, self.bbox_head.loss(example, preds)
        else:
            return boxes, bev_feature, voxel_feature, None 


@DETECTORS.register_module
class VoxelNetFusion(VoxelNet):
    """Deploy fusion network without focal module
    this model structure similar to mvxnet
    """
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        network2d=None,
        fusion=None,
    ):
        super(VoxelNetFusion, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        self.total_time = 0
        self.iters = 0
        assert network2d is not None and fusion is not None, 'network2d & fusion must be defined'
        self.network2d = builder.build_network2d(network2d)
        self.fusion = builder.build_fusion(fusion)

    def extract_feat(self, data, batch_dict, example):
        input_features = self.reader(data["features"], data["num_voxels"])
        x, voxel_feature = self.backbone(
            input_features, batch_dict, data["coors"], data["batch_size"], data["input_shape"], example,
            fuse_func=self.fusion
        )
        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def extract_feat2d(self, data):
        img_feature = {}
        for single_view in data.keys():
            single_result = self.network2d(data[single_view])
            for layer in single_result.keys():
                if layer not in img_feature:
                    img_feature[layer] = {}
                img_feature[layer][single_view] = single_result[layer]

        return img_feature

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )
        batch_dict = {}
        if self.network2d is not None and self.fusion is not None:
            batch_dict["images"] = example['cam']
            batch_dict['image_shape'] = example["image_shape"]
            batch_dict['calib'] = example['calib']
            batch_dict['img_feat'] = self.extract_feat2d(example['cam'])
            if 'aug_matrix_inv' in example:
                batch_dict['aug_matrix_inv'] = example['aug_matrix_inv']

        start_time = time.time()
        x, _ = self.extract_feat(data, batch_dict, example)
        preds = self.bbox_head(x)

        if return_loss:
            ret = self.bbox_head.loss(example, preds, batch_dict)
            return ret
        else:
            boxes = self.bbox_head.predict(example, preds, self.test_cfg)
            end_time = time.time()
            self.total_time += (end_time - start_time)
            self.iters += 1
            #print('avg_speed', self.total_time/self.iters)
            return boxes
