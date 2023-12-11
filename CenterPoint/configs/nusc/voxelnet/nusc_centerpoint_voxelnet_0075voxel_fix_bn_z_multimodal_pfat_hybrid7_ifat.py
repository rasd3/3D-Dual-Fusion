import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor

tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

voxel_generator = dict(
    range=[-54, -54, -5.0, 54, 54, 3.0],
    voxel_size=[0.075, 0.075, 0.2],
    max_points_in_voxel=10,
    max_voxel_num=[120000, 160000],
)

image_scale = 2/3
#  image_scale = 0.5
image_list = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
              'CAM_BACK',  'CAM_BACK_LEFT',  'CAM_BACK_RIGHT']
depth_thres = {'CAM_FRONT': 1,  'CAM_FRONT_LEFT': 0, 'CAM_FRONT_RIGHT': 0,
               'CAM_BACK': 0.5, 'CAM_BACK_LEFT': 0,  'CAM_BACK_RIGHT': 0}
# model settings
model = dict(
    type="VoxelNetFusion",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        # type='SimpleVoxel',
        num_input_features=5,
    ),
    backbone=dict(
        type="SpMiddleResNetFHDFusion", 
        num_input_features=5, 
        ds_factor=8,
        TOPK=True,
        USE_IMG=True,
        SKIP_LOSS=True,
    ),
    network2d=dict(
        type='PyramidFeat2D',
        optimize=True,
        ret_original=True,
        model_cfg=dict(
            name='SemDeepLabV3',
            backbone='ResNet50',
            num_class=21, # pretrained on COCO 
            args={"feat_extract_layer": ["layer1"],
                  "pretrained_path": "checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth"},
            channel_reduce={
                "in_channels": [256],
                "out_channels": [16],
                "kernel_size": [1],
                "stride": [1],
                "bias": [False]
            },
        ),
    ),
    fusion=dict(
        type='VoxelWithPointProjection',
        fuse_mode='pfat',
        pfat_cfg=dict(
            fusion_method='sum',
            feature_modal='hybrid',
            hybrid_cfg=dict(
                attn_layer='BiGateSum1D_2',
                q_method='sum',
                q_rep_place=['weight']
            ),
            num_bins=80,
            num_channels=[256],
            query_num_feat=128,
            num_enc_layers=2,
            max_num_ne_voxel=26000,
            pos_encode_method='depth'),
        lt_cfg=dict(
            npoint=2048,
            radius=2.0,
            nsample=32,
            num_layers=2,
            attn_feat_agg_method='unique',
            feat_agg_method='replace'
        ),
        interpolate=False,
        voxel_size=voxel_generator['voxel_size'],
        pc_range=voxel_generator['range'],
        image_list=image_list,
        image_scale=image_scale,
        depth_thres=depth_thres,
        ifat_cfg=dict(
            fusion_method='Basicgate_patch_iv_multivoxel',
            img_num_channel=256,
            pts_num_channel=128,
            voxel_feat_channel=[32,64,128],
            voxel_idx=[0,2] #x_conv2,x_conv3,x_conv4
        ),
    ),
    neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=sum([256, 256]),
        tasks=tasks,
        dataset='nuscenes',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel': (2, 2)},
        share_conv_channel=64,
        dcn_head=False
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)


train_cfg = dict(assigner=assigner)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2,
    ),
    score_threshold=0.1,
    pc_range=[-54, -54],
    out_size_factor=get_downsample_factor(model),
    voxel_size=[0.075, 0.075]
)

# dataset settings
dataset_type = "NuScenesDataset"
version= "v1.0-trainval"
nsweeps = 10
data_root = "data/nuScenes"
with_info = True

db_sampler = dict(
    type="GT-AUG",
    enable=True,
    db_info_path="data/nuScenes/dbinfos_train_10sweeps_withvelo.pkl",
    sample_groups=[
        dict(car=2),
        dict(truck=3),
        dict(construction_vehicle=7),
        dict(bus=4),
        dict(trailer=6),
        dict(barrier=2),
        dict(motorcycle=6),
        dict(bicycle=6),
        dict(pedestrian=2),
        dict(traffic_cone=2),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                car=5,
                truck=5,
                bus=5,
                trailer=5,
                construction_vehicle=5,
                traffic_cone=5,
                barrier=5,
                motorcycle=5,
                bicycle=5,
                pedestrian=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)
train_preprocessor = dict(
    mode="train",
    augmentation=['db_sample', 'flip', 'rotate', 'rescale', 'translate'], # NO Augmentation during Training, 'db_sample'
    sample_method='by_order',
    shuffle_points=True,
    global_rot_noise=[-0.78539816, 0.78539816],
    global_scale_noise=[0.9, 1.1],
    global_translate_std=0.5,
    db_sampler=db_sampler,
    class_names=class_names,
    with_info=with_info, # load
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    with_info=with_info, # load
)


train_pipeline = [
    dict(type="LoadPointCloudImageFromFile", dataset=dataset_type, image_scale=image_scale,
                                             image_list=image_list, with_info=with_info, 
                                             version=version, data_root=data_root),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
    # dict(type='PointCloudCollect', keys=['points', 'voxels', 'annotations', 'calib']),
]
test_pipeline = [
    dict(type="LoadPointCloudImageFromFile", dataset=dataset_type, image_scale=image_scale,
                                             image_list=image_list, with_info=with_info, 
                                             version=version, data_root=data_root),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = "data/nuScenes/infos_train_10sweeps_withvelo_filter_True.pkl"
#  train_anno = "data/nuScenes/infos_train_mini_1_7_10sweeps_withvelo_filter_True.pkl"
val_anno = "data/nuScenes/infos_val_10sweeps_withvelo_filter_True.pkl"
test_anno = None

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
        with_info=with_info,

    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        with_info=with_info, # load
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        with_info=with_info, # load
    ),
)



optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 20
device_ids = range(8)[:4]
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None 
workflow = [('train', 1)]
