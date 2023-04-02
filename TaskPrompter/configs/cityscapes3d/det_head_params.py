from easydict import EasyDict as edict
INF = 1e8

test_cfg = edict(
    use_rotate_nms=True,
    nms_across_levels=False,
    nms_pre= 1000, #-1
    nms_thr=0.3,
    score_thr=0.05,
    min_bbox_size=0,
    max_per_img=200,
)

group_reg_dims=(2, 1, 3, 3, 4)  # offset, depth, size, rot, bbox2d
reg_branch=(
    (256, ),  # offset
    (256, ),  # depth
    (256, ),  # size
    (256, ),  # rot
    (256, )  # bbox2d
)
num_classes=6
bbox_code_size=9  # For cityscapes

##### strides #####
strides = [8, 16, 32, 32, 64] # will be adjusted in config.py based on the actual input img resolution
fpn_scale_no = 5

regress_ranges = ((-1, 96), (96, 192), (192, 384),(384, 768), (768, INF)) # 2xRR
use_direction_classifier=True
pred_bbox2d=True
pred_keypoints=False

det_model_params = edict(
    num_classes=num_classes,
    regress_ranges=regress_ranges,
    center_sampling=True,
    center_sample_radius=1.5,
    norm_on_bbox=True,
    centerness_alpha=2.5,
    loss_cls=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=5.0),
    loss_dir=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    loss_bbox=dict(
        type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
    loss_centerness=dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        loss_weight=1.0),
    loss_bbox2d=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
    loss_consistency=dict(type='GIoULoss', loss_weight=1.0),
    stacked_convs=3,
    strides=strides,
    use_direction_classifier=use_direction_classifier,
    background_label=None,
    diff_rad_by_sin=True,
    dir_offset=0,
    bbox_code_size=bbox_code_size,  # For cityscapes
    pred_bbox2d=pred_bbox2d,
    pred_keypoints=pred_keypoints,
    group_reg_dims=group_reg_dims,  # offset, depth, size, rot*3
    code_weight=[1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0,
        1.0, 1.0, 1.0, 1.0],
    test_cfg=test_cfg,
)

neck_out_chan = 256
neck=dict(
    type='FPN',
    out_channels=neck_out_chan,
    start_level=0,
    add_extra_convs='on_output',
    num_outs=fpn_scale_no,
    relu_before_extra_convs=True)

det_head_params = edict(
    num_classes=num_classes,
    in_channels=neck_out_chan,
    centerness_on_reg=True,
    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
    dcn_on_last_conv=True,
    conv_bias=True,
    use_direction_classifier=use_direction_classifier,
    group_reg_dims=group_reg_dims, 
    reg_branch=reg_branch,
    centerness_branch=(256,),
    cls_branch=(256, 128,),
    dir_branch=(256,),
    fpn_scale_no=fpn_scale_no,
    feat_channels=256,
    stacked_convs=3,
    bbox_code_size=bbox_code_size,  # For cityscapes
    pred_bbox2d=pred_bbox2d,
    pred_keypoints=pred_keypoints,
    conv_cfg=None,
    init_cfg=None,
    neck_cfg=neck
)

def get_CS_metrics_of_interest(inp):
    out = {}
    out['mDetection_Score'] = inp['mDetection_Score']
    out['mAP'] = inp['mAP']
    out['car_Detection_Score'] = inp['Detection_Score']['car']
    out['car_AP'] = inp['AP']['car']['auc']
    out['car_OS_Yaw'] = inp['OS_Yaw']['car']['auc']
    out['car_OS_Pitch_Roll'] = inp['OS_Pitch_Roll']['car']['auc']
    out['car_Center_Dist'] = inp['Center_Dist']['car']['auc']
    out['car_Size_Similarity'] = inp['Size_Similarity']['car']['auc']
    return out