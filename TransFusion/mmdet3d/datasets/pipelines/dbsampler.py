import copy
import mmcv
import numpy as np
import os
import os.path as osp
from skimage import io

from mmdet3d.core.bbox import box_np_ops
from mmdet3d.datasets.pipelines import data_augment_utils
from mmdet.datasets import PIPELINES
from ..registry import OBJECTSAMPLERS

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

class BatchSampler:
    """Class for sampling specific category of ground truths.

    Args:
        sample_list (list[dict]): List of samples.
        name (str | None): The category of samples. Default: None.
        epoch (int | None): Sampling epoch. Default: None.
        shuffle (bool): Whether to shuffle indices. Default: False.
        drop_reminder (bool): Drop reminder. Default: False.
    """

    def __init__(self,
                 sampled_list,
                 name=None,
                 epoch=None,
                 shuffle=True,
                 drop_reminder=False):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        self._shuffle = shuffle
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        """Sample specific number of ground truths and return indices.

        Args:
            num (int): Sampled number.

        Returns:
            list[int]: Indices of sampled ground truths.
        """
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        """Reset the index of batchsampler to zero."""
        assert self._name is not None
        # print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        """Sample specific number of ground truths.

        Args:
            num (int): Sampled number.

        Returns:
            list[dict]: Sampled ground truths.
        """
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]


@OBJECTSAMPLERS.register_module()
class DataBaseSampler(object):
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        data_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str]): List of classes. Default: None.
        points_loader(dict): Config of points loader. Default: dict(
            type='LoadPointsFromFile', load_dim=4, use_dim=[0,1,2,3])
    """

    def __init__(self,
                 info_path,
                 data_root,
                 rate,
                 prepare,
                 sample_groups,
                 classes=None,
                 points_loader=dict(
                     type='LoadPointsFromFile',
                     coord_type='LIDAR',
                     load_dim=4,
                     use_dim=[0, 1, 2, 3])):
        super().__init__()
        self.data_root = data_root
        self.info_path = info_path
        self.rate = rate
        self.prepare = prepare
        self.classes = classes
        self.cat2label = {name: i for i, name in enumerate(classes)}
        self.label2cat = {i: name for i, name in enumerate(classes)}
        self.points_loader = mmcv.build_from_cfg(points_loader, PIPELINES)

        db_infos = mmcv.load(info_path)

        # filter database infos
        from mmdet3d.utils import get_root_logger
        logger = get_root_logger()
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos')
        for prep_func, val in prepare.items():
            db_infos = getattr(self, prep_func)(db_infos, val)
        logger.info('After filter database:')
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos')

        self.db_infos = db_infos

        # load sample groups
        # TODO: more elegant way to load sample groups
        self.sample_groups = []
        for name, num in sample_groups.items():
            self.sample_groups.append({name: int(num)})

        self.group_db_infos = self.db_infos  # just use db_infos
        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys())
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict = {}
        for k, v in self.group_db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)
        # TODO: No group_sampling currently

    @staticmethod
    def filter_by_difficulty(db_infos, removed_difficulty):
        """Filter ground truths by difficulties.

        Args:
            db_infos (dict): Info of groundtruth database.
            removed_difficulty (list): Difficulties that are not qualified.

        Returns:
            dict: Info of database after filtering.
        """
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
        return new_db_infos

    @staticmethod
    def filter_by_min_points(db_infos, min_gt_points_dict):
        """Filter ground truths by number of points in the bbox.

        Args:
            db_infos (dict): Info of groundtruth database.
            min_gt_points_dict (dict): Different number of minimum points
                needed for different categories of ground truths.

        Returns:
            dict: Info of database after filtering.
        """
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

    def sample_all(self, gt_bboxes, gt_labels, img=None, data_info=None, sample=None):
        """Sampling all categories of bboxes.

        Args:
            gt_bboxes (np.ndarray): Ground truth bounding boxes.
            gt_labels (np.ndarray): Ground truth labels of boxes.

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.

                - gt_labels_3d (np.ndarray): ground truths labels \
                    of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): \
                    sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            class_label = self.cat2label[class_name]
            # sampled_num = int(max_sample_num -
            #                   np.sum([n == class_name for n in gt_names]))
            sampled_num = int(max_sample_num -
                              np.sum([n == class_label for n in gt_labels]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes = []
        avoid_coll_boxes = gt_bboxes

        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                   avoid_coll_boxes)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]['box3d_lidar'][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls], axis=0)

                    sampled_gt_bboxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0)

        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
            #  sample_coords = box_np_ops.rbbox3d_to_corners(sampled_gt_bboxes)
            sample_coords = np.array(sample.new_box(sampled_gt_bboxes).corners)
            # center = sampled_gt_bboxes[:, 0:3]

            # num_sampled = len(sampled)
            s_points_list = []
            crop_imgs_list = []
            count = 0
            for _idx, info in enumerate(sampled):
                file_path = os.path.join(
                    self.data_root,
                    info['path']) if self.data_root else info['path']
                results = dict(pts_filename=file_path)
                s_points = self.points_loader(results)['points']
                s_points.translate(info['box3d_lidar'][:3])

                count += 1

                s_points_list.append(s_points)

                if data_info is not None:
                    # Transform points
                    if 'token' in info:
                        sample_record = data_info.get('sample', info['token'])
                    else:
                        sample_record = data_info.get('sample', info['path'].split('/')[-1].split('_')[0])
                    pointsensor_token = sample_record['data']['LIDAR_TOP']
                    crop_img = []
                    cam_orders = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
                            'CAM_BACK', 'CAM_BACK_LEFT']
                    for c_idx, _key in enumerate(cam_orders):
                        cam_key = _key.upper()
                        camera_token = sample_record['data'][cam_key]
                        cam = data_info.get('sample_data', camera_token)
                        lidar2cam, cam_intrinsic = get_lidar2cam_matrix(data_info, pointsensor_token, cam)
                        points_3d = np.concatenate([sample_coords[_idx], np.ones((len(sample_coords[_idx]), 1))], axis=-1)
                        # points_cam = points_3d @ lidar2cam.T
                        points_cam = lidar2cam @ points_3d.T
                        # Filter useless boxes according to depth
                        if not (points_cam[2,:]>0).all():
                            continue
                        point_img = view_points(points_cam[:3, :], np.array(cam_intrinsic), normalize=True)
                        point_img = point_img.transpose()[:,:2].astype(np.int64)
                        cam_img = get_image(osp.join(data_info.dataroot, cam['filename']))
                        minxy = np.min(point_img, axis=0)
                        maxxy = np.max(point_img, axis=0)
                        bbox = np.concatenate([minxy, maxxy], axis=0)
                        bbox[0::2] = np.clip(bbox[0::2], a_min=0, a_max=cam_img.shape[1]-1)
                        bbox[1::2] = np.clip(bbox[1::2], a_min=0, a_max=cam_img.shape[0]-1)
                        if (bbox[2]-bbox[0])*(bbox[3]-bbox[1])==0:
                            continue
                        crop_img = cam_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        break
                    crop_imgs_list.append(crop_img)

            gt_labels = np.array([self.cat2label[s['name']] for s in sampled],
                                 dtype=np.long)
            if 'token' in sampled[0]:
                sampled_gt_bboxes[:, 2] -= sampled_gt_bboxes[:, 5] / 2
            ret = {
                'gt_labels_3d':
                gt_labels,
                'gt_bboxes_3d':
                sampled_gt_bboxes,
                'points':
                s_points_list[0].cat(s_points_list),
                'group_ids':
                np.arange(gt_bboxes.shape[0],
                          gt_bboxes.shape[0] + len(sampled)),
                'img_crops': crop_imgs_list
            }

        return ret

    def sample_class_v2(self, name, num, gt_bboxes):
        """Sampling specific categories of bounding boxes.

        Args:
            name (str): Class of objects to be sampled.
            num (int): Number of sampled bboxes.
            gt_bboxes (np.ndarray): Ground truth boxes.

        Returns:
            list[dict]: Valid samples after collision test.
        """
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6])

        if False and 'token' in sampled[0].keys():
            # for focal dbinfo pkl
            for i in range(len(sampled)):
                sampled[i]['box3d_lidar'] = sampled[i]['box3d_lidar'][[0, 1, 2, 3, 4, 5, 8, 6, 7]]

        sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)

        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0]:]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples

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

def get_image(path):
    """
    Loads image for a sample. Copied from PCDet.
    Args:
        idx: int, Sample index
    Returns:
        image: (H, W, 3), RGB Image
    """
    assert osp.exists(path)
    image = io.imread(path)
    #  image = image.astype(np.float32)
    #  image /= 255.0
    return image
