# ------------------------------------------------------------------------------
# Mostly borrowed from MMDetection3D
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from termcolor import colored
#from .sync_bn.inplace_abn.bn import InPlaceABNSync

#BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)

# openmm
# from mmdet.models.builder import HEADS, build_loss
from .mm_builder import build_neck
from mmcv.cnn import ConvModule
# from mmcv.ops import ModulatedDeformConv2dPack
from mmcv.runner import BaseModule

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

INF = 1e8
import pdb



def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = functools.partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale

def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (torch.Tensor): The value to be converted.
        offset (float, optional): Offset to set the value range. \
            Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.

    Returns:
        torch.Tensor: Value in the range of \
            [-offset * period, (1-offset) * period]
    """
    return val - torch.floor(val / period + offset) * period

def xywhr2xyxyr(boxes_xywhr):
    """Convert a rotated boxes in XYWHR format to XYXYR format.

    Args:
        boxes_xywhr (torch.Tensor): Rotated boxes in XYWHR format.

    Returns:
        torch.Tensor: Converted boxes in XYXYR format.
    """
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[:, 2] / 2
    half_h = boxes_xywhr[:, 3] / 2

    boxes[:, 0] = boxes_xywhr[:, 0] - half_w
    boxes[:, 1] = boxes_xywhr[:, 1] - half_h
    boxes[:, 2] = boxes_xywhr[:, 0] + half_w
    boxes[:, 3] = boxes_xywhr[:, 1] + half_h
    boxes[:, 4] = boxes_xywhr[:, 4]
    return boxes

class FCOS3DHead(BaseModule):

    def __init__(self,
                 num_classes,
                 in_channels,
                 centerness_on_reg=True,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 centerness_branch=(64, ),
                feat_channels=256,
                stacked_convs=4,
                dcn_on_last_conv=False,
                conv_bias='auto',
                use_direction_classifier=True,
                group_reg_dims=(2, 1, 3, 1, 2),  # offset, depth, size, rot, velo,
                cls_branch=(128, 64),
                reg_branch=(
                    (128, 64),  # offset
                    (128, 64),  # depth
                    (64, ),  # size
                    (64, ),  # rot
                    ()  # velo
                ),
                fpn_scale_no=None,
                bbox_code_size=None,
                pred_bbox2d=True,
                pred_keypoints=False,
                dir_branch=(64, ),
                conv_cfg=None,
                init_cfg=None,
                neck_cfg=None):
        super(FCOS3DHead, self).__init__(init_cfg=init_cfg)
        self.neck = build_neck(neck_cfg)
        self.centerness_on_reg = centerness_on_reg
        self.centerness_branch = centerness_branch
        self.pred_attrs = False
        
        self.bbox_code_size = bbox_code_size
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.use_direction_classifier = use_direction_classifier
        self.group_reg_dims = list(group_reg_dims)
        self.cls_branch = cls_branch
        self.reg_branch = reg_branch
        assert len(reg_branch) == len(group_reg_dims), 'The number of '\
            'element in reg_branch and group_reg_dims should be the same.'
        self.out_channels = []
        for reg_branch_channels in reg_branch:
            if len(reg_branch_channels) > 0:
                self.out_channels.append(reg_branch_channels[-1])
            else:
                self.out_channels.append(-1)
        self.dir_branch = dir_branch
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.pred_bbox2d = pred_bbox2d
        self.pred_keypoints = pred_keypoints

        self.fpn_scale_no = fpn_scale_no

        self._init_layers()
        if init_cfg is None:
            self.init_cfg = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=dict(
                    type='Normal', name='conv_cls', std=0.01, bias_prob=0.01))

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()
        self.conv_centerness_prev = self._init_branch(
            conv_channels=self.centerness_branch,
            conv_strides=(1, ) * len(self.centerness_branch))
        self.conv_centerness = nn.Conv2d(self.centerness_branch[-1], 1, 1)
        self.scale_dim = 3
        if self.pred_bbox2d:
            self.scale_dim += 1
        if self.pred_keypoints:
            self.scale_dim += 1
        self.scales = nn.ModuleList([
            nn.ModuleList([Scale(1.0) for _ in range(self.scale_dim)]) for _ in range(self.fpn_scale_no)
        ])  # only for offset, depth and size regression

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            try:
                self.cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.conv_bias))
            except:
                pdb.set_trace()

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))


    def _init_branch(self, conv_channels=(64), conv_strides=(1)):
        """Initialize conv layers as a prediction branch."""
        conv_before_pred = nn.ModuleList()
        if isinstance(conv_channels, int):
            conv_channels = [self.feat_channels] + [conv_channels]
            conv_strides = [conv_strides]
        else:
            conv_channels = [self.feat_channels] + list(conv_channels)
            conv_strides = list(conv_strides)
        for i in range(len(conv_strides)):
            conv_before_pred.append(
                ConvModule(
                    conv_channels[i],
                    conv_channels[i + 1],
                    3,
                    stride=conv_strides[i],
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

        return conv_before_pred

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls_prev = self._init_branch(
            conv_channels=self.cls_branch,
            conv_strides=(1, ) * len(self.cls_branch))
        self.conv_cls = nn.Conv2d(self.cls_branch[-1], self.cls_out_channels,
                                  1)
        self.conv_reg_prevs = nn.ModuleList()
        self.conv_regs = nn.ModuleList()
        for i in range(len(self.group_reg_dims)):
            reg_dim = self.group_reg_dims[i]
            reg_branch_channels = self.reg_branch[i]
            out_channel = self.out_channels[i]
            if len(reg_branch_channels) > 0:
                self.conv_reg_prevs.append(
                    self._init_branch(
                        conv_channels=reg_branch_channels,
                        conv_strides=(1, ) * len(reg_branch_channels)))
                self.conv_regs.append(nn.Conv2d(out_channel, reg_dim, 1))
            else:
                self.conv_reg_prevs.append(None)
                self.conv_regs.append(
                    nn.Conv2d(self.feat_channels, reg_dim, 1))
        if self.use_direction_classifier:
            self.conv_dir_cls_prev = self._init_branch(
                conv_channels=self.dir_branch,
                conv_strides=(1, ) * len(self.dir_branch))
            self.conv_dir_cls = nn.Conv2d(self.dir_branch[-1], 2*3, 1) # dir_cls for all three euler angles

    def init_weights(self):
        for modules in [self.cls_convs, self.reg_convs, self.conv_cls_prev]:
            for m in modules:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        for conv_reg_prev in self.conv_reg_prevs:
            if conv_reg_prev is None:
                continue
            for m in conv_reg_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        if self.use_direction_classifier:
            for m in self.conv_dir_cls_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        if self.pred_attrs:
            for m in self.conv_attr_prev:
                if isinstance(m.conv, nn.Conv2d):
                    normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        for conv_reg in self.conv_regs:
            normal_init(conv_reg, std=0.01)
        if self.use_direction_classifier:
            normal_init(self.conv_dir_cls, std=0.01, bias=bias_cls)
        if self.pred_attrs:
            normal_init(self.conv_attr, std=0.01, bias=bias_cls)

        for m in self.conv_centerness_prev:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        normal_init(self.conv_centerness, std=0.01)

    def forward(self, feat):
        # get fpn
        feats = self.neck(feat) # pn neck

        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * bbox_code_size.
                dir_cls_preds (list[Tensor]): Box scores for direction class
                    predictions on each scale level, each is a 4D-tensor,
                    the channel number is num_points * 2. (bin = 2).
                attr_preds (list[Tensor]): Attribute scores for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * num_attrs.
                centernesses (list[Tensor]): Centerness for each scale level,
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           )

    def forward_single(self, x, scale):
        """Forward features of a single scale levle.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox and direction class \
                predictions, centerness predictions of input feature maps.
        """
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        # clone the cls_feat for reusing the feature map afterwards
        clone_cls_feat = cls_feat.clone()
        for conv_cls_prev_layer in self.conv_cls_prev:
            clone_cls_feat = conv_cls_prev_layer(clone_cls_feat)
        cls_score = self.conv_cls(clone_cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = []
        for i in range(len(self.group_reg_dims)):
            # clone the reg_feat for reusing the feature map afterwards
            clone_reg_feat = reg_feat.clone()
            if len(self.reg_branch[i]) > 0:
                for conv_reg_prev_layer in self.conv_reg_prevs[i]:
                    clone_reg_feat = conv_reg_prev_layer(clone_reg_feat)
            bbox_pred.append(self.conv_regs[i](clone_reg_feat))
        bbox_pred = torch.cat(bbox_pred, dim=1)

        dir_cls_pred = None
        if self.use_direction_classifier:
            clone_reg_feat = reg_feat.clone()
            for conv_dir_cls_prev_layer in self.conv_dir_cls_prev:
                clone_reg_feat = conv_dir_cls_prev_layer(clone_reg_feat)
            dir_cls_pred = self.conv_dir_cls(clone_reg_feat)


        # return cls_score, bbox_pred, dir_cls_pred, attr_pred, cls_feat, \
        #     reg_feat

        if self.centerness_on_reg:
            clone_reg_feat = reg_feat.clone()
            for conv_centerness_prev_layer in self.conv_centerness_prev:
                clone_reg_feat = conv_centerness_prev_layer(clone_reg_feat)
            centerness = self.conv_centerness(clone_reg_feat)
        else:
            clone_cls_feat = cls_feat.clone()
            for conv_centerness_prev_layer in self.conv_centerness_prev:
                clone_cls_feat = conv_centerness_prev_layer(clone_cls_feat)
            centerness = self.conv_centerness(clone_cls_feat)
        # scale the bbox_pred of different level
        # only apply to offset, depth and size prediction
        scale_offset, scale_depth, scale_size = scale[0:3]

        clone_bbox_pred = bbox_pred.clone()
        bbox_pred[:, :2] = scale_offset(clone_bbox_pred[:, :2]).float()
        bbox_pred[:, 2] = scale_depth(clone_bbox_pred[:, 2]).float()
        bbox_pred[:, 3:6] = scale_size(clone_bbox_pred[:, 3:6]).float()

        bbox_pred[:, 2] = bbox_pred[:, 2].exp()
        bbox_pred[:, 3:6] = bbox_pred[:, 3:6].exp() + 1e-6  # avoid size=0

        if self.pred_keypoints:
            scale_kpts = scale[3]
            # 2 dimension of offsets x 8 corners of a 3D bbox
            bbox_pred[:, self.bbox_code_size:self.bbox_code_size + 16] = \
                torch.tanh(scale_kpts(clone_bbox_pred[
                    :, self.bbox_code_size:self.bbox_code_size + 16]).float())

        if self.pred_bbox2d:
            scale_bbox2d = scale[-1]
            # The last four dimensions are offsets to four sides of a 2D bbox
            bbox_pred[:, -4:] = F.relu(scale_bbox2d(clone_bbox_pred[:, -4:]).float())

        return cls_score, bbox_pred, dir_cls_pred, centerness


def test_func():
    from easydict import EasyDict as edict
    p = edict(
            num_classes=10,
            in_channels=128,
            regress_ranges=((-1, 48), (48, 96), (96, 192), (192, 384),
                            (384, INF)),
            center_sampling=True,
            center_sample_radius=1.5,
            norm_on_bbox=True,
            centerness_on_reg=True,
            centerness_alpha=2.5,
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            loss_dir=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_attr=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_centerness=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0),
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            centerness_branch=(64, ),
        feat_channels=256,
        stacked_convs=4,
        strides=(4, 8, 16, 32, 64),
        dcn_on_last_conv=False,
        conv_bias='auto',
        background_label=None,
        use_direction_classifier=True,
        diff_rad_by_sin=True,
        dir_offset=0,
        bbox_code_size=9,  # For nuscenes
        pred_attrs=False,
        num_attrs=9,  # For nuscenes
        group_reg_dims=(2, 1, 3, 1, 2),  # offset, depth, size, rot, velo,
        cls_branch=(128, 64),
        reg_branch=(
            (128, 64),  # offset
            (128, 64),  # depth
            (64, ),  # size
            (64, ),  # rot
            ()  # velo
        ),
        dir_branch=(64, ),
        attr_branch=(64, ),
        conv_cfg=None,
        test_cfg=None,
        init_cfg=None)

    det_head = FCOS3DHead(**p)

    inp = [torch.rand((2, 128, 256, 256)), torch.rand((2, 128, 128, 128)), torch.rand((2, 128, 64, 64)), torch.rand((2, 128, 32, 32))]
    out = det_head(inp)

if __name__ == '__main__':
    test_func()