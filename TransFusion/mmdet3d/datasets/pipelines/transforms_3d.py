import numpy as np
from mmcv import is_tuple_of
from mmcv.utils import build_from_cfg

from mmdet3d.core import VoxelGenerator
from mmdet3d.core.bbox import box_np_ops
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import RandomFlip
from ..registry import OBJECTSAMPLERS
from .data_augment_utils import noise_per_object_v3_

import cv2
from pyquaternion import Quaternion

try:
    from nuscenes import NuScenes
    from nuscenes.utils import splits
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import transform_matrix
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
except:
    print("nuScenes devkit not Found!")

@PIPELINES.register_module()
class RandomFlip3D(RandomFlip):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                 sync_2d=True,
                 flip_ratio_bev_horizontal=0.0,
                 flip_ratio_bev_vertical=0.0,
                 **kwargs):
        super(RandomFlip3D, self).__init__(
            flip_ratio=flip_ratio_bev_horizontal, **kwargs)
        self.sync_2d = sync_2d
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Default: horizontal.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are \
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](
                np.array([], dtype=np.float32))
        assert len(input_dict['bbox3d_fields']) == 1
        for key in input_dict['bbox3d_fields']:
            input_dict['points'] = input_dict[key].flip(
                direction, points=input_dict['points'])

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and \
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction', \
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added \
                into result dict.
        """
        # filp 2D image and its annotations
        #  super(RandomFlip3D, self).__call__(input_dict)

        if self.sync_2d:
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio else False
                input_dict['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in input_dict:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(sync_2d={},'.format(self.sync_2d)
        repr_str += 'flip_ratio_bev_vertical={})'.format(
            self.flip_ratio_bev_vertical)
        return repr_str


@PIPELINES.register_module()
class ObjectSample(object):
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(self, db_sampler, sample_2d=False, with_info=False):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)
        self.with_info = with_info
        self.keep_raw = True
        self.sample_method = 'by_order'

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (np.ndarray): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        if self.with_info:
            data_info = input_dict["data_info"]
            cam_images = input_dict['img']
            cam_orders = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
                    'CAM_BACK', 'CAM_BACK_LEFT']
        else:
            data_info, cam_images = None, None

        num_gt = len(gt_bboxes_3d)
        crop_img_list = [[np.zeros((1,1,1))] * num_gt for _ in range(len(cam_orders))]
        if self.with_info:
            sample_record = data_info.get('sample', input_dict['sample_idx'])
            pointsensor_token = sample_record['data']['LIDAR_TOP']
        if self.with_info and num_gt:
            # Transform points
            sample_coords = input_dict["gt_bboxes_3d"].corners
            #  sample_coords = box_np_ops.rbbox3d_to_corners(gt_bboxes_3d.tensor.cpu().numpy())
            #  crop_img_list = [[] for _ in range(len(sample_coords))]
            #  crop_img_list = [[np.zeros((1,1,1))] * len(sample_coords) for _ in range(len(cam_orders))]
            # Crop images from raw images
            for _idx, _key in enumerate(cam_orders):
                cam_key = _key.upper()
                camera_token = sample_record['data'][cam_key]
                cam = data_info.get('sample_data', camera_token)
                lidar2cam, cam_intrinsic = get_lidar2cam_matrix(data_info, pointsensor_token, cam)
                points_3d = np.concatenate([sample_coords, np.ones((*sample_coords.shape[:2], 1))], axis=-1)
                points_cam = (points_3d @ lidar2cam.T).T
                # Filter useless boxes according to depth
                cam_mask = (points_cam[2] > 0).all(axis=0)
                cam_count = cam_mask.nonzero()[0]
                if cam_mask.sum() == 0:
                    continue
                points_cam = points_cam[...,cam_mask].reshape(4, -1)
                point_img = view_points(points_cam[:3, :], np.array(cam_intrinsic), normalize=True)
                point_img = point_img.reshape(3, 8, -1)
                point_img = point_img.transpose()[...,:2]
                minxy = np.min(point_img, axis=-2)
                maxxy = np.max(point_img, axis=-2)
                bbox = np.concatenate([minxy, maxxy], axis=-1)
                #  bbox = (bbox * res['image_scale']).astype(np.int32)
                bbox[:,0::2] = np.clip(bbox[:,0::2], a_min=0, a_max=input_dict['img_shape'][1]-1)
                bbox[:,1::2] = np.clip(bbox[:,1::2], a_min=0, a_max=input_dict['img_shape'][0]-1)
                cam_mask = (bbox[:,2]-bbox[:,0])*(bbox[:,3]-bbox[:,1])>0
                if cam_mask.sum() == 0:
                    continue
                cam_count = cam_count[cam_mask]
                bbox = bbox[cam_mask]

                for __idx, _box in enumerate(bbox.astype(np.int)):
                    #  crop_img_list[__idx] = cam_images[_idx][_box[1]:_box[3],_box[0]:_box[2]]
                    crop_img_list[_idx][cam_count[__idx]] = cam_images[_idx][_box[1]:_box[3],_box[0]:_box[2]]

        # change to float for blending operation
        points = input_dict['points']
        if self.sample_2d:
            img = input_dict['img']
            # Assume for now 3D & 2D bboxes are the same
            #  sampled_dict = self.db_sampler.sample_all(
                #  gt_bboxes_3d.tensor.numpy(),
                #  gt_labels_3d,
                #  img=img,
                #  data_info=data_info
                #  )
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                img=img,
                data_info=data_info,
                sample=gt_bboxes_3d
                )
        else:
            #  sampled_dict = self.db_sampler.sample_all(
                #  gt_bboxes_3d.tensor.numpy(), gt_labels_3d, img=None)
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, img=None, sample=gt_bboxes_3d)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            # for changed pkl
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            points = points.cat([sampled_points, points])

            if self.with_info:
                # Transform points
                sample_coords = gt_bboxes_3d.corners
                #  sample_coords = box_np_ops.rbbox3d_to_corners(sampled_dict['gt_bboxes_3d'])
                raw_gt_box_num = len(sampled_dict["gt_bboxes_3d"])
                #  sample_crops = crop_img_list + sampled_dict['img_crops']
                sample_crops = []
                for cam_order in range(len(cam_orders)):
                    sample_crops.append(crop_img_list[cam_order] + sampled_dict['img_crops'])
                if not self.keep_raw:
                    points_coords = points[:,:4].copy()
                    points_coords[:,-1] = 1
                    point_keep_mask = np.ones(len(points_coords)).astype(np.bool)
                # Paste image according to sorted strategy
                for _idx, _key in enumerate(cam_orders):
                    cam_key = _key.upper()
                    camera_token = sample_record['data'][cam_key]
                    cam = data_info.get('sample_data', camera_token)
                    lidar2cam, cam_intrinsic = get_lidar2cam_matrix(data_info, pointsensor_token, cam)
                    points_3d = np.concatenate([sample_coords, np.ones((*sample_coords.shape[:2], 1))], axis=-1)
                    points_cam = (points_3d @ lidar2cam.T).T
                    depth = points_cam[2]
                    cam_mask = (depth > 0).all(axis=0)
                    cam_count = cam_mask.nonzero()[0]
                    if cam_mask.sum() == 0:
                        continue
                    depth = depth.mean(0)[cam_mask]
                    points_cam = points_cam[...,cam_mask].reshape(4, -1)
                    point_img = view_points(points_cam[:3, :], np.array(cam_intrinsic), normalize=True)
                    point_img = point_img.reshape(3, 8, -1)
                    point_img = point_img.transpose()[...,:2]
                    minxy = np.min(point_img, axis=-2)
                    maxxy = np.max(point_img, axis=-2)
                    bbox = np.concatenate([minxy, maxxy], axis=-1)
                    #  bbox = (bbox * res['image_scale']).astype(np.int32)
                    bbox[:,0::2] = np.clip(bbox[:,0::2], a_min=0, a_max=input_dict['img_shape'][1]-1)
                    bbox[:,1::2] = np.clip(bbox[:,1::2], a_min=0, a_max=input_dict['img_shape'][0]-1)
                    cam_mask = (bbox[:,2]-bbox[:,0])*(bbox[:,3]-bbox[:,1])>0
                    if cam_mask.sum() == 0:
                        continue
                    depth = depth[cam_mask]
                    if 'depth' in self.sample_method:
                        paste_order = depth.argsort()
                        paste_order = paste_order[::-1]
                    else:
                        paste_order = np.arange(len(depth), dtype=np.int64)
                    cam_count = cam_count[cam_mask][paste_order]
                    bbox = bbox[cam_mask][paste_order]

                    paste_mask = -255 * np.ones(input_dict['img_shape'][:2], dtype=np.int64)
                    fg_mask = np.zeros(input_dict['img_shape'][:2], dtype=np.int64)
                    for _count, _box in zip(cam_count, bbox.astype(np.int)):
                        img_crop = sample_crops[_idx][_count]
                        #  img_crop = sample_crops[_count]
                        if len(img_crop) == 0: continue
                        if img_crop.shape[0] == 0 or img_crop.shape[1] == 0: continue
                        if _box[2] - _box[0] < 1 or _box[3] - _box[1] < 0: continue
                        img_crop = cv2.resize(img_crop, tuple(_box[[2,3]]-_box[[0,1]]))
                        cam_images[_idx][_box[1]:_box[3],_box[0]:_box[2]] = img_crop
                        paste_mask[_box[1]:_box[3],_box[0]:_box[2]] = _count
                        # foreground area of original point cloud in image plane
                        if _count < raw_gt_box_num:
                            fg_mask[_box[1]:_box[3],_box[0]:_box[2]] = 1
                    # calculate modify mask
                    if not self.keep_raw:
                        points_cam = (points_coords @ lidar2cam.T).T
                        cam_mask = points_cam[2] > 0
                        if cam_mask.sum() == 0:
                            continue
                        point_img = view_points(points_cam[:3, :], np.array(cam_intrinsic), normalize=True)
                        point_img = point_img.transpose()[...,:2]
                        #  point_img = (point_img * res['image_scale']).astype(np.int64)
                        cam_mask = (point_img[:,0] > 0) & (point_img[:,0] < input_dict['img_shape'][1]) & \
                                   (point_img[:,1] > 0) & (point_img[:,1] < input_dict['img_shape'][0]) & cam_mask
                        point_img = point_img[cam_mask]
                        new_mask = paste_mask[point_img[:,1], point_img[:,0]]==(points_idx[cam_mask]+raw_gt_box_num)
                        raw_fg = (fg_mask == 1) & (paste_mask >= 0) & (paste_mask < raw_gt_box_num)
                        raw_bg = (fg_mask == 0) & (paste_mask < 0)
                        raw_mask = raw_fg[point_img[:,1], point_img[:,0]] | raw_bg[point_img[:,1], point_img[:,0]]
                        keep_mask = new_mask | raw_mask
                        point_keep_mask[cam_mask] = point_keep_mask[cam_mask] & keep_mask

                # Replace the original images
                input_dict['img'] = cam_images
                # remove overlaped LIDAR points
                if not self.keep_raw:
                    points = points[point_keep_mask]
            if self.sample_2d and False:
                sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)

                input_dict['gt_bboxes'] = gt_bboxes_2d
                input_dict['img'] = sampled_dict['img']

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.long)
        input_dict['points'] = points

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str


@PIPELINES.register_module()
class ObjectNoise(object):
    """Apply noise to each GT objects in the scene.

    Args:
        translation_std (list[float], optional): Standard deviation of the
            distribution where translation noise are sampled from.
            Defaults to [0.25, 0.25, 0.25].
        global_rot_range (list[float], optional): Global rotation to the scene.
            Defaults to [0.0, 0.0].
        rot_range (list[float], optional): Object rotation range.
            Defaults to [-0.15707963267, 0.15707963267].
        num_try (int, optional): Number of times to try if the noise applied is
            invalid. Defaults to 100.
    """

    def __init__(self,
                 translation_std=[0.25, 0.25, 0.25],
                 global_rot_range=[0.0, 0.0],
                 rot_range=[-0.15707963267, 0.15707963267],
                 num_try=100):
        self.translation_std = translation_std
        self.global_rot_range = global_rot_range
        self.rot_range = rot_range
        self.num_try = num_try

    def __call__(self, input_dict):
        """Call function to apply noise to each ground truth in the scene.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after adding noise to each object, \
                'points', 'gt_bboxes_3d' keys are updated in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        points = input_dict['points']

        # TODO: check this inplace function
        numpy_box = gt_bboxes_3d.tensor.numpy()
        numpy_points = points.tensor.numpy()

        noise_per_object_v3_(
            numpy_box,
            numpy_points,
            rotation_perturb=self.rot_range,
            center_noise_std=self.translation_std,
            global_random_rot_range=self.global_rot_range,
            num_try=self.num_try)

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d.new_box(numpy_box)
        input_dict['points'] = points.new_point(numpy_points)
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(num_try={},'.format(self.num_try)
        repr_str += ' translation_std={},'.format(self.translation_std)
        repr_str += ' global_rot_range={},'.format(self.global_rot_range)
        repr_str += ' rot_range={})'.format(self.rot_range)
        return repr_str


@PIPELINES.register_module()
class GlobalRotScaleTrans(object):
    """Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of ranslation
            noise. This apply random translation to a scene by a noise, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0]
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 shift_height=False):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std
        self.shift_height = shift_height

    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        if not isinstance(self.translation_std, (list, tuple, np.ndarray)):
            translation_std = [
                self.translation_std, self.translation_std,
                self.translation_std
            ]
        else:
            translation_std = self.translation_std
        translation_std = np.array(translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        input_dict['points'].translate(trans_factor)
        input_dict['pcd_trans'] = trans_factor
        for key in input_dict['bbox3d_fields']:
            input_dict[key].translate(trans_factor)

    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        rotation = self.rot_range
        if not isinstance(rotation, list):
            rotation = [-rotation, rotation]
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        for key in input_dict['bbox3d_fields']:
            if len(input_dict[key].tensor) != 0:
                points, rot_mat_T = input_dict[key].rotate(
                    noise_rotation, input_dict['points'])
                input_dict['points'] = points
                input_dict['pcd_rotation'] = rot_mat_T
        # input_dict['points_instance'].rotate(noise_rotation)

    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']
        points = input_dict['points']
        points.scale(scale)
        if self.shift_height:
            assert 'height' in points.attribute_dims.keys()
            points.tensor[:, points.attribute_dims['height']] *= scale
        input_dict['points'] = points

        for key in input_dict['bbox3d_fields']:
            input_dict[key].scale(scale)

    def _random_scale(self, input_dict):
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor' are updated \
                in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and \
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(rot_range={},'.format(self.rot_range)
        repr_str += ' scale_ratio_range={},'.format(self.scale_ratio_range)
        repr_str += ' translation_std={})'.format(self.translation_std)
        repr_str += ' shift_height={})'.format(self.shift_height)
        return repr_str


@PIPELINES.register_module()
class PointShuffle(object):
    """Shuffle input points."""

    def __call__(self, input_dict):
        """Call function to shuffle points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points' keys are updated \
                in the result dict.
        """
        input_dict['points'].shuffle()
        return input_dict

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class ObjectRangeFilter(object):
    """Filter objects by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)
        self.bev_range = self.pcd_range[[0, 1, 3, 4]]

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        mask = gt_bboxes_3d.in_range_bev(self.bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(point_cloud_range={})'.format(self.pcd_range.tolist())
        return repr_str


@PIPELINES.register_module()
class PointsRangeFilter(object):
    """Filter points by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points' keys are updated \
                in the result dict.
        """
        points = input_dict['points']
        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        input_dict['points'] = clean_points
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(point_cloud_range={})'.format(self.pcd_range.tolist())
        return repr_str


@PIPELINES.register_module()
class ObjectNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str


@PIPELINES.register_module()
class IndoorPointSample(object):
    """Indoor point sample.

    Sampling data to a certain number.

    Args:
        name (str): Name of the dataset.
        num_points (int): Number of points to be sampled.
    """

    def __init__(self, num_points):
        self.num_points = num_points

    def points_random_sampling(self,
                               points,
                               num_samples,
                               replace=None,
                               return_choices=False):
        """Points random sampling.

        Sample points to a certain number.

        Args:
            points (np.ndarray): 3D Points.
            num_samples (int): Number of samples to be sampled.
            replace (bool): Whether the sample is with or without replacement.
            Defaults to None.
            return_choices (bool): Whether return choice. Defaults to False.

        Returns:
            tuple[np.ndarray] | np.ndarray:

                - points (np.ndarray): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        if replace is None:
            replace = (points.shape[0] < num_samples)
        choices = np.random.choice(
            points.shape[0], num_samples, replace=replace)
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]

    def __call__(self, results):
        """Call function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = results['points']
        points, choices = self.points_random_sampling(
            points, self.num_points, return_choices=True)

        pts_instance_mask = results.get('pts_instance_mask', None)
        pts_semantic_mask = results.get('pts_semantic_mask', None)
        results['points'] = points

        if pts_instance_mask is not None and pts_semantic_mask is not None:
            pts_instance_mask = pts_instance_mask[choices]
            pts_semantic_mask = pts_semantic_mask[choices]
            results['pts_instance_mask'] = pts_instance_mask
            results['pts_semantic_mask'] = pts_semantic_mask

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(num_points={})'.format(self.num_points)
        return repr_str


@PIPELINES.register_module()
class BackgroundPointsFilter(object):
    """Filter background points near the bounding box.

    Args:
        bbox_enlarge_range (tuple[float], float): Bbox enlarge range.
    """

    def __init__(self, bbox_enlarge_range):
        assert (is_tuple_of(bbox_enlarge_range, float)
                and len(bbox_enlarge_range) == 3) \
            or isinstance(bbox_enlarge_range, float), \
            f'Invalid arguments bbox_enlarge_range {bbox_enlarge_range}'

        if isinstance(bbox_enlarge_range, float):
            bbox_enlarge_range = [bbox_enlarge_range] * 3
        self.bbox_enlarge_range = np.array(
            bbox_enlarge_range, dtype=np.float32)[np.newaxis, :]

    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points' keys are updated \
                in the result dict.
        """
        points = input_dict['points']
        gt_bboxes_3d = input_dict['gt_bboxes_3d']

        gt_bboxes_3d_np = gt_bboxes_3d.tensor.numpy()
        gt_bboxes_3d_np[:, :3] = gt_bboxes_3d.gravity_center.numpy()
        enlarged_gt_bboxes_3d = gt_bboxes_3d_np.copy()
        enlarged_gt_bboxes_3d[:, 3:6] += self.bbox_enlarge_range
        points_numpy = points.tensor.numpy()
        foreground_masks = box_np_ops.points_in_rbbox(points_numpy,
                                                      gt_bboxes_3d_np)
        enlarge_foreground_masks = box_np_ops.points_in_rbbox(
            points_numpy, enlarged_gt_bboxes_3d)
        foreground_masks = foreground_masks.max(1)
        enlarge_foreground_masks = enlarge_foreground_masks.max(1)
        valid_masks = ~np.logical_and(~foreground_masks,
                                      enlarge_foreground_masks)

        input_dict['points'] = points[valid_masks]
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[valid_masks]

        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[valid_masks]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(bbox_enlarge_range={})'.format(
            self.bbox_enlarge_range.tolist())
        return repr_str


@PIPELINES.register_module()
class VoxelBasedPointSampler(object):
    """Voxel based point sampler.

    Apply voxel sampling to multiple sweep points.

    Args:
        cur_sweep_cfg (dict): Config for sampling current points.
        prev_sweep_cfg (dict): Config for sampling previous points.
        time_dim (int): Index that indicate the time dimention
            for input points.
    """

    def __init__(self, cur_sweep_cfg, prev_sweep_cfg=None, time_dim=3):
        self.cur_voxel_generator = VoxelGenerator(**cur_sweep_cfg)
        self.cur_voxel_num = self.cur_voxel_generator._max_voxels
        self.time_dim = time_dim
        if prev_sweep_cfg is not None:
            assert prev_sweep_cfg['max_num_points'] == \
                cur_sweep_cfg['max_num_points']
            self.prev_voxel_generator = VoxelGenerator(**prev_sweep_cfg)
            self.prev_voxel_num = self.prev_voxel_generator._max_voxels
        else:
            self.prev_voxel_generator = None
            self.prev_voxel_num = 0

    def _sample_points(self, points, sampler, point_dim):
        """Sample points for each points subset.

        Args:
            points (np.ndarray): Points subset to be sampled.
            sampler (VoxelGenerator): Voxel based sampler for
                each points subset.
            point_dim (int): The dimention of each points

        Returns:
            np.ndarray: Sampled points.
        """
        voxels, coors, num_points_per_voxel = sampler.generate(points)
        if voxels.shape[0] < sampler._max_voxels:
            padding_points = np.zeros([
                sampler._max_voxels - voxels.shape[0], sampler._max_num_points,
                point_dim
            ],
                                      dtype=points.dtype)
            padding_points[:] = voxels[0]
            sample_points = np.concatenate([voxels, padding_points], axis=0)
        else:
            sample_points = voxels

        return sample_points

    def __call__(self, results):
        """Call function to sample points from multiple sweeps.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = results['points']
        original_dim = points.shape[1]

        # TODO: process instance and semantic mask while _max_num_points
        # is larger than 1
        # Extend points with seg and mask fields
        map_fields2dim = []
        start_dim = original_dim
        points_numpy = points.tensor.numpy()
        extra_channel = [points_numpy]
        for idx, key in enumerate(results['pts_mask_fields']):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(results[key][..., None])

        start_dim += len(results['pts_mask_fields'])
        for idx, key in enumerate(results['pts_seg_fields']):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(results[key][..., None])

        points_numpy = np.concatenate(extra_channel, axis=-1)

        # Split points into two part, current sweep points and
        # previous sweeps points.
        # TODO: support different sampling methods for next sweeps points
        # and previous sweeps points.
        cur_points_flag = (points_numpy[:, self.time_dim] == 0)
        cur_sweep_points = points_numpy[cur_points_flag]
        prev_sweeps_points = points_numpy[~cur_points_flag]
        if prev_sweeps_points.shape[0] == 0:
            prev_sweeps_points = cur_sweep_points

        # Shuffle points before sampling
        np.random.shuffle(cur_sweep_points)
        np.random.shuffle(prev_sweeps_points)

        cur_sweep_points = self._sample_points(cur_sweep_points,
                                               self.cur_voxel_generator,
                                               points_numpy.shape[1])
        if self.prev_voxel_generator is not None:
            prev_sweeps_points = self._sample_points(prev_sweeps_points,
                                                     self.prev_voxel_generator,
                                                     points_numpy.shape[1])

            points_numpy = np.concatenate(
                [cur_sweep_points, prev_sweeps_points], 0)
        else:
            points_numpy = cur_sweep_points

        if self.cur_voxel_generator._max_num_points == 1:
            points_numpy = points_numpy.squeeze(1)
        results['points'] = points.new_point(points_numpy[..., :original_dim])

        # Restore the correspoinding seg and mask fields
        for key, dim_index in map_fields2dim:
            results[key] = points_numpy[..., dim_index]

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""

        def _auto_indent(repr_str, indent):
            repr_str = repr_str.split('\n')
            repr_str = [' ' * indent + t + '\n' for t in repr_str]
            repr_str = ''.join(repr_str)[:-1]
            return repr_str

        repr_str = self.__class__.__name__
        indent = 4
        repr_str += '(\n'
        repr_str += ' ' * indent + f'num_cur_sweep={self.cur_voxel_num},\n'
        repr_str += ' ' * indent + f'num_prev_sweep={self.prev_voxel_num},\n'
        repr_str += ' ' * indent + f'time_dim={self.time_dim},\n'
        repr_str += ' ' * indent + 'cur_voxel_generator=\n'
        repr_str += f'{_auto_indent(repr(self.cur_voxel_generator), 8)},\n'
        repr_str += ' ' * indent + 'prev_voxel_generator=\n'
        repr_str += f'{_auto_indent(repr(self.prev_voxel_generator), 8)})'
        return repr_str


def get_lidar2cam_matrix(nusc, sensor_token, cam):
    # Get nuScenes lidar2cam and cam_intrinsic matrix
    pointsensor = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    ego_from_lidar = transform_matrix(
        cs_record["translation"], Quaternion(cs_record["rotation"]), inverse=False
    )

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    global_from_ego = transform_matrix(
        poserecord["translation"], Quaternion(poserecord["rotation"]), inverse=False
    )

    lidar2global = np.dot(global_from_ego, ego_from_lidar)

    # Transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])

    ego_from_global = transform_matrix(
        poserecord["translation"], Quaternion(poserecord["rotation"]), inverse=True
    )
    
    # Transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    cam_from_ego = transform_matrix(
        cs_record["translation"], Quaternion(cs_record["rotation"]), inverse=True
    )
    global2cam = np.dot(cam_from_ego, ego_from_global)
    lidar2cam = np.dot(global2cam, lidar2global)
    cam_intrinsic = cs_record['camera_intrinsic']

    return lidar2cam, cam_intrinsic

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    # copied from https://github.com/nutonomy/nuscenes-devkit/
    # only for debug use
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points
