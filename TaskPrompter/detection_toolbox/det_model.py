# Adapted by Hanrong Ye for TaskPrompter
# Base code is from MMDetection3D

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

from detection_toolbox.det_losses import SmoothL1Loss, CrossEntropyLoss, FocalLoss, GIoULoss
from detection_toolbox.det_tools import bbox3d2result, limit_period, xywhrst2xyxyrst, bbox_bev, xywhpra2xyxya, decode_yaw
from .det_tools import box3d_multiclass_nms, distance2bbox, bbox2result, points_cam2img, points_img2cam
import pdb

from mmdet3d.core.bbox import CameraInstance3DBoxes


INF = 1e8
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


class DetModel(nn.Module):
    '''Modified from FCOS3D
    '''
    def __init__(self,
                num_classes,
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
                loss_centerness=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0),
                 loss_dir=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox2d=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_consistency=dict(type='GIoULoss', loss_weight=1.0),
                stacked_convs=4,
                strides=(4, 8, 16, 32, 64),
                conv_bias='auto',
                background_label=None,
                use_direction_classifier=True,
                diff_rad_by_sin=True,
                dir_offset=0,
                bbox_code_size=9,  # For cityscapes
                pred_bbox2d=False,
                pred_keypoints=False,
                group_reg_dims=(2, 1, 3, 1, 2),  # offset, depth, size, rot, velo,
                code_weight=None,
                test_cfg=None,
                ):
        super().__init__()
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.centerness_alpha = centerness_alpha

        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.stacked_convs = stacked_convs
        self.strides = strides
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.use_direction_classifier = use_direction_classifier
        self.diff_rad_by_sin = diff_rad_by_sin
        self.dir_offset = dir_offset
        self.loss_cls = self.build_loss(loss_cls)
        self.loss_bbox = self.build_loss(loss_bbox)
        self.loss_centerness = self.build_loss(loss_centerness)
        self.loss_dir = self.build_loss(loss_dir)
        if pred_bbox2d:
            self.loss_bbox2d = self.build_loss(loss_bbox2d)
            self.loss_consistency = self.build_loss(loss_consistency)
        self.bbox_code_size = bbox_code_size
        self.group_reg_dims = list(group_reg_dims)
        self.code_weight = code_weight
        self.pred_bbox2d = pred_bbox2d
        self.pred_keypoints = pred_keypoints
        if self.pred_keypoints:
            self.kpts_start = 9 

        self.out_channels = []
        self.fp16_enabled = False
        self.background_label = (
            num_classes if background_label is None else background_label)
        # background_label should be either 0 or num_classes
        assert (self.background_label == 0
                or self.background_label == num_classes)
        self.test_cfg = test_cfg

    @staticmethod
    def build_loss(loss_params):
        if loss_params['type'] == 'SmoothL1Loss':
            del loss_params['type']
            loss = SmoothL1Loss(**loss_params)
        elif loss_params['type'] == 'CrossEntropyLoss':
            del loss_params['type']
            loss = CrossEntropyLoss(**loss_params)
        elif loss_params['type'] == 'FocalLoss':
            del loss_params['type']
            loss = FocalLoss(**loss_params)
        elif loss_params['type'] == 'GIoULoss':
            del loss_params['type']
            loss = GIoULoss(**loss_params)
        else:
            raise NotImplementedError
        return loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2): # using sine addition formula
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1 (torch.Tensor): Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2 (torch.Tensor): Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            tuple[torch.Tensor]: ``boxes1`` and ``boxes2`` whose 7th \
                dimensions are changed.
        """
        # all euler angles
        rad_pred_encoding = torch.sin(boxes1[..., 6:9]) * torch.cos(
            boxes2[..., 6:9])
        rad_tg_encoding = torch.cos(boxes1[..., 6:9]) * torch.sin(boxes2[...,
                                                                         6:9])
        boxes1 = torch.cat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 9:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 9:]],
                           dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(reg_targets,
                             dir_offset=0,
                             num_bins=2,
                             one_hot=True):
        """Encode direction to 0 ~ num_bins-1.

        Args:
            reg_targets (torch.Tensor): Bbox regression targets.
            dir_offset (int): Direction offset.
            num_bins (int): Number of bins to divide 2*PI.
            one_hot (bool): Whether to encode as one hot.

        Returns:
            torch.Tensor: Encoded direction targets.
        """
        dir_cls_targets_list = []
        for rot in range(6, 9):
            rot_gt = reg_targets[..., rot] # yaw
            offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
            dir_cls_targets = torch.floor(offset_rot /
                                        (2 * np.pi / num_bins)).long()
            dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
            if one_hot:
                dir_targets = torch.zeros(
                    *list(dir_cls_targets.shape),
                    num_bins,
                    dtype=reg_targets.dtype,
                    device=dir_cls_targets.device)
                dir_targets.scatter_(dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
                dir_cls_targets = dir_targets
            dir_cls_targets_list.append(dir_cls_targets)
        return dir_cls_targets_list

    def fetch_preds(self, preds_list, denorm_on_bbox=False):
        lvl_no = len(preds_list)
        bbox_preds_list = []
        centerness_list = []
        dir_cls_list = []
        cls_scores_list = []

        for lvl in range(lvl_no):
            preds = preds_list[lvl]
            # L, W, H have to be positive so we need to clamp them
            bbox_preds = preds[:, :9, :, :] # delta_x, delta_y, z, l, w, h, pitch, roll, yaw

            bbox_preds[:, 2] = bbox_preds[:, 2] #.exp()
            bbox_preds[:, 3:6] = bbox_preds[:, 3:6] + 1e-6 #.exp() + 1e-6  # avoid size=0

            if self.norm_on_bbox:
                if denorm_on_bbox:
                    # Note that this line is conducted only when testing
                    bbox_preds[:, :2] *= self.strides[lvl]

            bbox_preds_list.append(bbox_preds) # [(B, 10, H, W) * lvl_no]
            centerness_list.append(preds[:, 9:10, :, :]) # 10
            dir_cls_list.append(preds[:, 10:12, :, :]) # 11, 12
            cls_scores_list.append(preds[:, 12:, :, :]) # 6 classes

        return bbox_preds_list, centerness_list, cls_scores_list, dir_cls_list

    def denorm_on_bbox(self, bbox_preds_list):

        lvl_no = len(bbox_preds_list)
        denorm_bbox_preds_list = []

        for lvl in range(lvl_no):
            bbox_preds = bbox_preds_list[lvl].clone()
            bbox_preds[:, :2] *= self.strides[lvl]
            if self.pred_bbox2d:
                bbox_preds[:, -4:] *= self.strides[lvl]
            if self.pred_keypoints:
                max_regress_range = self.strides[lvl] * self.regress_ranges[0][1] / \
                    self.strides[0]
                bbox_preds[
                    :, self.bbox_code_size:self.bbox_code_size + 16] *= \
                        max_regress_range

            denorm_bbox_preds_list.append(bbox_preds) # [(B, 10, H, W) * lvl_no]

        return denorm_bbox_preds_list


    def loss(self, preds, labels):
        """
        preds: HxWxC, C=19: 2-offset, 1-depth, 3-dims, 3-rotation, 1-centerness, 9-labels
        
        """
        cls_scores, bbox_preds, dir_cls_preds, centernesses = preds

        bs = len(labels['det_labels'])
        gt_bboxes = [] # [(no_bbox_per_sample, 2) * batch_size]
        gt_labels = [] # [(no_bbox_per_sample,) * batch_size]
        gt_bboxes_3d = []
        gt_labels_3d = gt_labels
        centers2d = [] # [(no_bbox_per_sample, 2) * batch_size]
        depths = [] # [(no_bbox_per_sample, 1) * batch_size]
        gt_bboxes_ignore = None
        img_metas = []

        loss_sum = torch.tensor(0, dtype=cls_scores[0].dtype, device=cls_scores[0].device)
        n_i = 0
        for _i in range(bs):
            if labels['det_label_number'][_i] != 0: 
                gt_bboxes.append(labels['det_labels'][_i]['bbox_modal'])
                gt_labels.append(labels['det_labels'][_i]['label'])
                gt_bboxes_3d.append(torch.cat([labels['det_labels'][_i]['center_S'], 
                                        labels['det_labels'][_i]['size_S'], 
                                        labels['det_labels'][_i]['rotation_S']], dim=1))
                gt_labels_3d = gt_labels
                centers2d.append(labels['det_labels'][_i]['center_I'][:, :2])
                depths.append(labels['det_labels'][_i]['center_I'][:, 2])
                img_metas.append({k: v[_i] for k, v in labels['meta'].items()})
                n_i += 1
            else:
                for sca in range(len(cls_scores)):
                    try:
                        # we have to use the useless predition tensor to contribute to the loss to make pytorch happy.
                        loss_sum += cls_scores[sca][n_i].sum() * 0
                        loss_sum += bbox_preds[sca][n_i].sum() * 0
                        loss_sum += dir_cls_preds[sca][n_i].sum() * 0
                        loss_sum += centernesses[sca][n_i].sum() * 0

                        # remove prediction for samples without labels
                        cls_scores[sca] = torch.cat([cls_scores[sca][:n_i], cls_scores[sca][n_i+1:]])
                        bbox_preds[sca] = torch.cat([bbox_preds[sca][:n_i], bbox_preds[sca][n_i+1:]])
                        dir_cls_preds[sca] = torch.cat([dir_cls_preds[sca][:n_i], dir_cls_preds[sca][n_i+1:]])
                        centernesses[sca] = torch.cat([centernesses[sca][:n_i], centernesses[sca][n_i+1:]])
                    except:
                        pdb.set_trace()
        if len(gt_bboxes) == 0:
            loss_dict = {}
            return loss_dict, loss_sum
            

        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            centernesses (list[Tensor]): Centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_3d (list[Tensor]): 3D boxes ground truth with shape of
                (num_gts, code_size).
            gt_labels_3d (list[Tensor]): same as gt_labels
            centers2d (list[Tensor]): 2D centers on the image with shape of
                (num_gts, 2).
            depths (list[Tensor]): Depth ground truth with shape of
                (num_gts, ).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses) 
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels_3d, bbox_targets_3d, centerness_targets = \
            self.get_targets(
                all_level_points, gt_bboxes, gt_labels, gt_bboxes_3d,
                gt_labels_3d, centers2d, depths)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds, dir_cls_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, sum(self.group_reg_dims))
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_dir_cls_preds = [
            dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2*3) # dir_cls for all three euler angles
            for dir_cls_pred in dir_cls_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_dir_cls_preds = torch.cat(flatten_dir_cls_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels_3d = torch.cat(labels_3d)
        flatten_bbox_targets_3d = torch.cat(bbox_targets_3d)
        flatten_centerness_targets = torch.cat(centerness_targets)
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels_3d >= 0)
                    & (flatten_labels_3d < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_dict = dict()

        loss_dict['loss_cls'] = self.loss_cls(
            flatten_cls_scores,
            flatten_labels_3d,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_dir_cls_preds = flatten_dir_cls_preds[pos_inds]
        if num_pos > 0:
            pos_bbox_targets_3d = flatten_bbox_targets_3d[pos_inds]
            pos_centerness_targets = flatten_centerness_targets[pos_inds]
            pos_points = flatten_points[pos_inds]
            bbox_weights = pos_centerness_targets.new_ones(
                len(pos_centerness_targets), sum(self.group_reg_dims))
            equal_weights = pos_centerness_targets.new_ones(
                pos_centerness_targets.shape)

            code_weight = self.code_weight
            if code_weight:
                assert len(code_weight) == sum(self.group_reg_dims)
                bbox_weights = bbox_weights * bbox_weights.new_tensor(
                    code_weight)

            if self.use_direction_classifier:
                pos_dir_cls_targets = self.get_direction_target(
                    pos_bbox_targets_3d, self.dir_offset, one_hot=False)

            if self.diff_rad_by_sin:
                pos_bbox_preds, pos_bbox_targets_3d = self.add_sin_difference(
                    pos_bbox_preds, pos_bbox_targets_3d)

            loss_dict['loss_offset'] = self.loss_bbox(
                pos_bbox_preds[:, :2],
                pos_bbox_targets_3d[:, :2],
                weight=bbox_weights[:, :2],
                avg_factor=equal_weights.sum())
            loss_dict['loss_depth'] = self.loss_bbox(
                pos_bbox_preds[:, 2],
                pos_bbox_targets_3d[:, 2],
                weight=bbox_weights[:, 2],
                avg_factor=equal_weights.sum())
            loss_dict['loss_size'] = self.loss_bbox(
                pos_bbox_preds[:, 3:6],
                pos_bbox_targets_3d[:, 3:6],
                weight=bbox_weights[:, 3:6],
                avg_factor=equal_weights.sum())
            loss_dict['loss_rotsin'] = self.loss_bbox(
                pos_bbox_preds[:, 6:9],
                pos_bbox_targets_3d[:, 6:9],
                weight=bbox_weights[:, 6:9],
                avg_factor=equal_weights.sum())

            proj_bbox2d_inputs = (bbox_preds, pos_dir_cls_preds, labels_3d,
                                  bbox_targets_3d, pos_points, pos_inds,
                                  img_metas)

            # direction classification loss
            if self.use_direction_classifier:
                loss_dict['loss_dir'] = 0
                for rot in range(3):
                    loss_dict['loss_dir'] += self.loss_dir(
                        pos_dir_cls_preds[:, rot*2:(rot+1)*2],
                        pos_dir_cls_targets[rot],
                        equal_weights,
                        avg_factor=equal_weights.sum())


            loss_dict['loss_centerness'] = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)

            if self.pred_keypoints:
                # use smoothL1 to compute consistency loss for keypoints
                # normalize the offsets with strides
                proj_bbox2d_preds, pos_decoded_bbox2d_preds, kpts_targets = \
                    self.get_proj_bbox2d(*proj_bbox2d_inputs, with_kpts=True)
                loss_dict['loss_kpts'] = self.loss_bbox(
                    pos_bbox_preds[:, self.kpts_start:self.kpts_start + 16],
                    kpts_targets,
                    weight=bbox_weights[:,
                                        self.kpts_start:self.kpts_start + 16],
                    avg_factor=equal_weights.sum())

            # 2d bbox consistency loss
            if self.pred_bbox2d:
                loss_dict['loss_bbox2d'] = self.loss_bbox2d(
                    pos_bbox_preds[:, -4:],
                    pos_bbox_targets_3d[:, -4:],
                    weight=bbox_weights[:, -4:],
                    avg_factor=equal_weights.sum())

        else:
            loss_dict['loss_offset'] = pos_bbox_preds[:, :2].sum()
            loss_dict['loss_size'] = pos_bbox_preds[:, 3:6].sum()
            loss_dict['loss_rotsin'] = pos_bbox_preds[:, 6:9].sum()
            loss_dict['loss_depth'] = pos_bbox_preds[:, 2].sum()
            if self.pred_bbox2d:
                loss_dict['loss_bbox2d'] = pos_bbox_preds[:, -4:].sum()
            loss_dict['loss_centerness'] = pos_centerness.sum()
            if self.use_direction_classifier:
                loss_dict['loss_dir'] = pos_dir_cls_preds.sum()

        loss_sum += sum(l for l in loss_dict.values())

        return loss_dict, loss_sum

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   dir_cls_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on each scale level, each is a 4D-tensor,
                the channel number is num_points * 2. (bin = 2)
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds) == len(centernesses) 
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            if self.use_direction_classifier:
                dir_cls_pred_list = [
                    dir_cls_preds[i][img_id].detach()
                    for i in range(num_levels)
                ]
            else:
                dir_cls_pred_list = [
                    cls_scores[i][img_id].new_full(
                        [2*3, *cls_scores[i][img_id].shape[1:]], 0).detach()
                    for i in range(num_levels)
                ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            input_meta = img_metas[img_id]
            det_bboxes = self._get_bboxes_single(
                cls_score_list, bbox_pred_list, dir_cls_pred_list,
                centerness_pred_list, mlvl_points, input_meta,
                cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           dir_cls_preds,
                           centernesses,
                           mlvl_points,
                           input_meta,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * bbox_code_size, H, W).
            dir_cls_preds (list[Tensor]): Box scores for direction class
                predictions on a single scale level with shape \
                (num_points * 2, H, W)
            attr_preds (list[Tensor]): Attribute scores for each scale level
                Has shape (N, num_points * num_attrs, H, W)
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 2).
            input_meta (dict): Metadata of input image.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            tuples[Tensor]: Predicted 3D boxes, scores, labels and attributes.
        """
        view = input_meta['K_matrix']
        scale_factor = input_meta['scale_factor']
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_centers2d = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_centerness = []
        mlvl_bboxes2d = None
        if self.pred_bbox2d:
            mlvl_bboxes2d = []

        for cls_score, bbox_pred, dir_cls_pred, centerness, points \
                          in zip(cls_scores, bbox_preds, dir_cls_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 3, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1,
                                                     sum(self.group_reg_dims))
            bbox_pred3d = bbox_pred[:, :self.bbox_code_size]
            if self.pred_bbox2d:
                bbox_pred2d = bbox_pred[:, -4:]
            nms_pre = cfg.nms_pre
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred3d = bbox_pred3d[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_pred = dir_cls_pred[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]
                centerness = centerness[topk_inds]
                if self.pred_bbox2d:
                    bbox_pred2d = bbox_pred2d[topk_inds, :]
            # change the offset to actual center predictions
            bbox_pred3d[:, :2] = points - bbox_pred3d[:, :2]
            if rescale:
                bbox_pred3d[:, :2] /= bbox_pred3d[:, :2].new_tensor(scale_factor.unsqueeze(0))
                if self.pred_bbox2d:
                    bbox_pred2d /= bbox_pred2d.new_tensor(scale_factor)
            pred_center2d = bbox_pred3d[:, :3].clone()
            bbox_pred3d[:, :3] = points_img2cam(bbox_pred3d[:, :3], view)
            mlvl_centers2d.append(pred_center2d)
            mlvl_bboxes.append(bbox_pred3d)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_centerness.append(centerness)
            if self.pred_bbox2d:
                bbox_pred2d = distance2bbox(
                    points, bbox_pred2d, max_shape=input_meta['img_size'])
                mlvl_bboxes2d.append(bbox_pred2d)

        mlvl_centers2d = torch.cat(mlvl_centers2d)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)
        if self.pred_bbox2d:
            mlvl_bboxes2d = torch.cat(mlvl_bboxes2d)

        if mlvl_bboxes.shape[0] > 0:
            for rot_i, rot in enumerate(range(6, 9)): # all euler angles
                dir_rot = limit_period(mlvl_bboxes[..., rot] - self.dir_offset, 0,
                                    np.pi) # this is becoz fcos3d uses direction classification so it only needs to predict angle in range (0, pi)
                mlvl_bboxes[..., rot] = (
                    dir_rot + self.dir_offset +
                    np.pi * mlvl_dir_scores[..., rot_i].to(mlvl_bboxes.dtype))

        mlvl_bboxes_for_nms = xywhpra2xyxya(bbox_bev(mlvl_bboxes)) # only condisder yaw in nms now

        mlvl_scores = torch.cat(mlvl_scores)
        if True: # padding for nms. nms requires the last class to be background
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        # no scale_factors in box3d_multiclass_nms
        # Then we multiply it from outside
        mlvl_nms_scores = mlvl_scores * mlvl_centerness[:, None]
        bboxes, scores, labels, centers2d, dir_scores, bboxes2d = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms, # only condisder yaw in nms now
                                    mlvl_nms_scores, mlvl_centers2d, cfg.score_thr,
                                    cfg.max_per_img, cfg, mlvl_dir_scores, mlvl_bboxes2d=mlvl_bboxes2d
                                    )

        outputs = (bboxes, scores, labels, centers2d,)
        if self.pred_bbox2d:
            bboxes2d = torch.cat([bboxes2d, scores[:, None]], dim=1)
            outputs = outputs + (bboxes2d, )

        return outputs

    @staticmethod
    def pts2Dto3D(points, view):
        
        """
        Args:
            points (torch.Tensor): points in 2D images, [N, 3], \
                3 corresponds with x, y in the image and depth.
            view (np.ndarray): camera instrinsic, [3, 3]

        Returns:
            torch.Tensor: points in 3D space. [N, 3], \
                3 corresponds with x, y, z in 3D space.
        """
        assert view.shape[0] <= 4
        assert view.shape[1] <= 4
        assert points.shape[1] == 3

        points2D = points[:, :2]
        depths = points[:, 2].view(-1, 1)
        unnorm_points2D = torch.cat([points2D * depths, depths], dim=1)

        viewpad = torch.eye(4, dtype=points2D.dtype, device=points2D.device)
        viewpad[:view.shape[0], :view.shape[1]] = points2D.new_tensor(view)
        inv_viewpad = torch.inverse(viewpad).transpose(0, 1)

        # Do operation in homogenous coordinates.
        nbr_points = unnorm_points2D.shape[0]
        homo_points2D = torch.cat(
            [unnorm_points2D,
             points2D.new_ones((nbr_points, 1))], dim=1)
        points3D = torch.mm(homo_points2D, inv_viewpad)[:, :3]

        return points3D

    def get_points(self, featmap_sizes, dtype, device, flatten=False):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self._get_points_single(featmap_sizes[i], self.strides[i],
                                        dtype, device, flatten))
        return mlvl_points

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        h, w = featmap_size
        x_range = torch.arange(w, dtype=dtype, device=device)
        y_range = torch.arange(h, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()

        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list,
                    gt_bboxes_3d_list, gt_labels_3d_list, centers2d_list,
                    depths_list):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            gt_bboxes_3d_list (list[Tensor]): 3D Ground truth bboxes of each
                image, each has shape (num_gt, bbox_code_size).
            gt_labels_3d_list (list[Tensor]): 3D Ground truth labels of each
                box, each has shape (num_gt,).
            centers2d_list (list[Tensor]): Projected 3D centers onto 2D image,
                each has shape (num_gt, 2).
            depths_list (list[Tensor]): Depth of projected 3D centers onto 2D
                image, each has shape (num_gt, 1).
            attr_labels_list (list[Tensor]): Attribute labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        _, bbox_targets_list, labels_3d_list, bbox_targets_3d_list, \
                centerness_targets_list = multi_apply(
                self._get_target_single,
                gt_bboxes_list,
                gt_labels_list,
                gt_bboxes_3d_list,
                gt_labels_3d_list,
                centers2d_list,
                depths_list,
                points=concat_points,
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points)

        # split to per img, per level
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        labels_3d_list = [
            labels_3d.split(num_points, 0) for labels_3d in labels_3d_list
        ] # [[(num_points_per_level,) x level_num] x image_number]
        bbox_targets_3d_list = [
            bbox_targets_3d.split(num_points, 0)
            for bbox_targets_3d in bbox_targets_3d_list
        ] # [[(num_points_per_level, 9) x level_num] x img_no]
        centerness_targets_list = [
            centerness_targets.split(num_points, 0)
            for centerness_targets in centerness_targets_list
        ] # [[(num_points_per_level, ) x level_num] x img_no]

        # concat all sample images at different levels
        concat_lvl_labels_3d = []
        concat_lvl_bbox_targets_3d = []
        concat_lvl_centerness_targets = []
        for i in range(num_levels):
            concat_lvl_labels_3d.append(
                torch.cat([labels[i] for labels in labels_3d_list]))
            concat_lvl_centerness_targets.append(
                torch.cat([
                    centerness_targets[i]
                    for centerness_targets in centerness_targets_list
                ]))
            bbox_targets_3d = torch.cat([
                bbox_targets_3d[i] for bbox_targets_3d in bbox_targets_3d_list
            ])
            if self.pred_bbox2d:
                bbox_targets = torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list])
                bbox_targets_3d = torch.cat([bbox_targets_3d, bbox_targets],
                                            dim=1)
            if self.norm_on_bbox:
                bbox_targets_3d[:, :2] = bbox_targets_3d[:, :2] / self.strides[i]
                if self.pred_bbox2d:
                    bbox_targets_3d[:, -4:] = \
                        bbox_targets_3d[:, -4:] / self.strides[i]
            concat_lvl_bbox_targets_3d.append(bbox_targets_3d)
        return concat_lvl_labels_3d, concat_lvl_bbox_targets_3d, \
            concat_lvl_centerness_targets

    def _get_target_single(self, gt_bboxes, gt_labels, gt_bboxes_3d,
                           gt_labels_3d, centers2d, depths,
                           points, regress_ranges, num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if not isinstance(gt_bboxes_3d, torch.Tensor):
            gt_bboxes_3d = gt_bboxes_3d.tensor.to(gt_bboxes.device)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.background_label), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_labels_3d.new_full(
                       (num_points,), self.background_label), \
                   gt_bboxes_3d.new_zeros((num_points, self.bbox_code_size)), \
                   gt_bboxes_3d.new_zeros((num_points,)), \

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        centers2d = centers2d[None].expand(num_points, num_gts, 2)
        gt_bboxes_3d = gt_bboxes_3d[None].expand(num_points, num_gts,
                                                 self.bbox_code_size)
        depths = depths[None, :, None].expand(num_points, num_gts, 1)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        delta_xs = (xs - centers2d[..., 0])[..., None] # on input image scale
        delta_ys = (ys - centers2d[..., 1])[..., None]
        bbox_targets_3d = torch.cat(
            (delta_xs, delta_ys, depths, gt_bboxes_3d[..., 3:]), dim=-1) # (num_points, num_gts, 9)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1) # (num_points, num_gts, 4)

        assert self.center_sampling is True, 'Setting center_sampling to '\
            'False has not been implemented for FCOS3D.'
        # condition1: inside a `center bbox`
        radius = self.center_sample_radius
        center_xs = centers2d[..., 0]
        center_ys = centers2d[..., 1]
        center_gts = torch.zeros_like(gt_bboxes)
        stride = center_xs.new_zeros(center_xs.shape)

        # project the points on current lvl back to the `original` sizes
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
            lvl_begin = lvl_end

        center_gts[..., 0] = center_xs - stride
        center_gts[..., 1] = center_ys - stride
        center_gts[..., 2] = center_xs + stride
        center_gts[..., 3] = center_ys + stride

        cb_dist_left = xs - center_gts[..., 0]
        cb_dist_right = center_gts[..., 2] - xs
        cb_dist_top = ys - center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys
        center_bbox = torch.stack(
            (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

        # condition2: limit the regression range for each location yhr: it is depending on the 2d bbox position
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # center-based criterion to deal with ambiguity
        dists = torch.sqrt(torch.sum(bbox_targets_3d[..., :2]**2, dim=-1))
        dists[inside_gt_bbox_mask == 0] = INF
        dists[inside_regress_range == 0] = INF
        min_dist, min_dist_inds = dists.min(dim=1)

        labels = gt_labels[min_dist_inds]
        labels_3d = gt_labels_3d[min_dist_inds] # (num_points,) class of each point
        labels[min_dist == INF] = self.background_label  # set as BG
        labels_3d[min_dist == INF] = self.background_label  # set as BG

        bbox_targets = bbox_targets[range(num_points), min_dist_inds] # (num_points, 4)
        bbox_targets_3d = bbox_targets_3d[range(num_points), min_dist_inds] # (num_points, 9)
        relative_dists = torch.sqrt(
            torch.sum(bbox_targets_3d[..., :2]**2,
                      dim=-1)) / (1.414 * stride[:, 0]) # square root of normal dist
        # [N, 1] / [N, 1]
        centerness_targets = torch.exp(-self.centerness_alpha * relative_dists) # (num_points,)

        return labels, bbox_targets, labels_3d, bbox_targets_3d, \
            centerness_targets

    ######################## Evaluation Kits ##########################
    def get_results_from_bbox(self, preds, label, rescale=False):
        """Adapted from simple_test
        Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        bs = len(label['meta']['img_name'])
        img_metas = [{k: v[s] for k, v in label['meta'].items()} for s in range(bs) ]

        # bbox_preds, centernesses, cls_scores, dir_cls_preds = self.fetch_preds(preds, denorm_on_bbox=True)
        cls_scores, bbox_preds, dir_cls_preds, centernesses = preds
        if self.norm_on_bbox:
            bbox_preds = self.denorm_on_bbox(bbox_preds)

        bbox_outputs = self.get_bboxes(
            cls_scores, bbox_preds, dir_cls_preds, centernesses, img_metas, cfg=self.test_cfg, rescale=rescale)

        if self.pred_bbox2d:
            bbox2d_img = [
                bbox2result(bboxes2d, labels, self.num_classes)
                for bboxes, scores, labels, centers2d, bboxes2d in bbox_outputs
            ]
            # bbox_outputs = [bbox_outputs[0][:-1]] # one sample per batch
            bbox_outputs = [_[:-1] for _ in bbox_outputs]

        bbox_img = [
            bbox3d2result(bboxes, scores, labels, centers2d)
            for bboxes, scores, labels, centers2d in bbox_outputs
        ]

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox
        if self.pred_bbox2d:
            for result_dict, img_bbox2d in zip(bbox_list, bbox2d_img):
                result_dict['img_bbox2d'] = img_bbox2d
        return bbox_list


    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        feats = self.extract_feats(imgs)

        # only support aug_test for one sample
        outs_list = [self.bbox_head(x) for x in feats]
        for i, img_meta in enumerate(img_metas):
            if img_meta[0]['pcd_horizontal_flip']:
                for j in range(len(outs_list[i])):  # for each prediction
                    if outs_list[i][j][0] is None:
                        continue
                    for k in range(len(outs_list[i][j])):
                        # every stride of featmap
                        outs_list[i][j][k] = torch.flip(
                            outs_list[i][j][k], dims=[3])
                reg = outs_list[i][1]
                for reg_feat in reg:
                    # offset_x
                    reg_feat[:, 0, :, :] = 1 - reg_feat[:, 0, :, :]
                    # velo_x
                    if self.bbox_head.pred_velo:
                        reg_feat[:, 7, :, :] = -reg_feat[:, 7, :, :]
                    # rotation
                    reg_feat[:, 6, :, :] = -reg_feat[:, 6, :, :] + np.pi

        merged_outs = []
        for i in range(len(outs_list[0])):  # for each prediction
            merged_feats = []
            for j in range(len(outs_list[0][i])):
                if outs_list[0][i][0] is None:
                    merged_feats.append(None)
                    continue
                # for each stride of featmap
                avg_feats = torch.mean(
                    torch.cat([x[i][j] for x in outs_list]),
                    dim=0,
                    keepdim=True)
                if i == 1:  # regression predictions
                    # rot/velo/2d det keeps the original
                    avg_feats[:, 6:, :, :] = \
                        outs_list[0][i][j][:, 6:, :, :]
                if i == 2:
                    # dir_cls keeps the original
                    avg_feats = outs_list[0][i][j]
                merged_feats.append(avg_feats)
            merged_outs.append(merged_feats)
        merged_outs = tuple(merged_outs)

        bbox_outputs = self.bbox_head.get_bboxes(
            *merged_outs, img_metas[0], rescale=rescale)
        if self.bbox_head.pred_bbox2d:
            from mmdet.core import bbox2result
            bbox2d_img = [
                bbox2result(bboxes2d, labels, self.bbox_head.num_classes)
                for bboxes, scores, labels, attrs, bboxes2d in bbox_outputs
            ]
            bbox_outputs = [bbox_outputs[0][:-1]]

        bbox_img = [
            bbox3d2result(bboxes, scores, labels, attrs)
            for bboxes, scores, labels, attrs in bbox_outputs
        ]

        bbox_list = dict()
        bbox_list.update(img_bbox=bbox_img[0])
        if self.bbox_head.pred_bbox2d:
            bbox_list.update(img_bbox2d=bbox2d_img[0])

        return [bbox_list]

    def get_proj_bbox2d(self,
                        bbox_preds,
                        pos_dir_cls_preds,
                        labels_3d,
                        bbox_targets_3d,
                        pos_points,
                        pos_inds,
                        img_metas,
                        pos_depth_cls_preds=None,
                        pos_weights=None,
                        pos_cls_scores=None,
                        with_kpts=False):
        """Decode box predictions and get projected 2D attributes.

        Args:
            bbox_preds (list[Tensor]): Box predictions for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            pos_dir_cls_preds (Tensor): Box scores for direction class
                predictions of positive boxes on all the scale levels in shape
                (num_pos_points, 2).
            labels_3d (list[Tensor]): 3D box category labels for each scale
                level, each is a 4D-tensor.
            bbox_targets_3d (list[Tensor]): 3D box targets for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * bbox_code_size.
            pos_points (Tensor): Foreground points.
            pos_inds (Tensor): Index of foreground points from flattened
                tensors.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            pos_depth_cls_preds (Tensor, optional): Probabilistic depth map of
                positive boxes on all the scale levels in shape
                (num_pos_points, self.num_depth_cls). Defaults to None.
            pos_weights (Tensor, optional): Location-aware weights of positive
                boxes in shape (num_pos_points, self.weight_dim). Defaults to
                None.
            pos_cls_scores (Tensor, optional): Classification scores of
                positive boxes in shape (num_pos_points, self.num_classes).
                Defaults to None.
            with_kpts (bool, optional): Whether to output keypoints targets.
                Defaults to False.

        Returns:
            tuple[Tensor]: Exterior 2D boxes from projected 3D boxes,
                predicted 2D boxes and keypoint targets (if necessary).
        """
        views = [np.array(img_meta['K_matrix']) for img_meta in img_metas]
        num_imgs = len(img_metas)
        img_idx = []
        for label in labels_3d:
            for idx in range(num_imgs):
                img_idx.append(
                    labels_3d[0].new_ones(int(len(label) / num_imgs)) * idx)
        img_idx = torch.cat(img_idx)
        pos_img_idx = img_idx[pos_inds]

        flatten_strided_bbox_preds = []
        flatten_strided_bbox2d_preds = []
        flatten_bbox_targets_3d = []
        flatten_strides = []

        for stride_idx, bbox_pred in enumerate(bbox_preds):
            flatten_bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(
                -1, sum(self.group_reg_dims))
            flatten_bbox_pred[:, :2] *= self.strides[stride_idx]
            flatten_bbox_pred[:, -4:] *= self.strides[stride_idx]
            flatten_strided_bbox_preds.append(
                flatten_bbox_pred[:, :self.bbox_code_size])
            flatten_strided_bbox2d_preds.append(flatten_bbox_pred[:, -4:])

            bbox_target_3d = bbox_targets_3d[stride_idx].clone()
            bbox_target_3d[:, :2] *= self.strides[stride_idx]
            bbox_target_3d[:, -4:] *= self.strides[stride_idx]
            flatten_bbox_targets_3d.append(bbox_target_3d)

            flatten_stride = flatten_bbox_pred.new_ones(
                *flatten_bbox_pred.shape[:-1], 1) * self.strides[stride_idx]
            flatten_strides.append(flatten_stride)

        flatten_strided_bbox_preds = torch.cat(flatten_strided_bbox_preds)
        flatten_strided_bbox2d_preds = torch.cat(flatten_strided_bbox2d_preds)
        flatten_bbox_targets_3d = torch.cat(flatten_bbox_targets_3d)
        flatten_strides = torch.cat(flatten_strides)
        pos_strided_bbox_preds = flatten_strided_bbox_preds[pos_inds]
        pos_strided_bbox2d_preds = flatten_strided_bbox2d_preds[pos_inds]
        pos_bbox_targets_3d = flatten_bbox_targets_3d[pos_inds]
        pos_strides = flatten_strides[pos_inds]

        pos_decoded_bbox2d_preds = distance2bbox(pos_points,
                                                 pos_strided_bbox2d_preds)

        pos_strided_bbox_preds[:, :2] = \
            pos_points - pos_strided_bbox_preds[:, :2]
        pos_bbox_targets_3d[:, :2] = \
            pos_points - pos_bbox_targets_3d[:, :2]

        box_corners_in_image = pos_strided_bbox_preds.new_zeros(
            (*pos_strided_bbox_preds.shape[:-1], 8, 2))
        box_corners_in_image_gt = pos_strided_bbox_preds.new_zeros(
            (*pos_strided_bbox_preds.shape[:-1], 8, 2))

        for idx in range(num_imgs):
            mask = (pos_img_idx == idx)
            if pos_strided_bbox_preds[mask].shape[0] == 0:
                continue
            cam2img = torch.eye(
                4,
                dtype=pos_strided_bbox_preds.dtype,
                device=pos_strided_bbox_preds.device)
            view_shape = views[idx].shape
            cam2img[:view_shape[0], :view_shape[1]] = \
                pos_strided_bbox_preds.new_tensor(views[idx])

            centers2d_preds = pos_strided_bbox_preds.clone()[mask, :2]
            centers2d_targets = pos_bbox_targets_3d.clone()[mask, :2]
            centers3d_targets = points_img2cam(pos_bbox_targets_3d[mask, :3],
                                               views[idx])

            # use predicted depth to re-project the 2.5D centers
            pos_strided_bbox_preds[mask, :3] = points_img2cam(
                pos_strided_bbox_preds[mask, :3], views[idx])
            pos_bbox_targets_3d[mask, :3] = centers3d_targets

            # depth fixed when computing re-project 3D bboxes
            pos_strided_bbox_preds[mask, 2] = \
                pos_bbox_targets_3d.clone()[mask, 2]

            # decode yaws
            if self.use_direction_classifier:
                pos_dir_cls_pred = pos_dir_cls_preds[mask].reshape(-1, 3, 2)
                pos_dir_cls_scores = torch.max(
                    pos_dir_cls_pred, dim=-1)[1]
                pdb.set_trace()
                pos_strided_bbox_preds[mask] = decode_yaw(
                    pos_strided_bbox_preds[mask], centers2d_preds,
                    pos_dir_cls_scores, self.dir_offset, cam2img)
            corners = CameraInstance3DBoxes(
                pos_strided_bbox_preds[mask],
                box_dim=self.bbox_code_size,
                origin=(0.5, 0.5, 0.5)).corners
            box_corners_in_image[mask] = points_cam2img(corners, cam2img)

            corners_gt = CameraInstance3DBoxes(
                pos_bbox_targets_3d[mask, :self.bbox_code_size],
                box_dim=self.bbox_code_size,
                origin=(0.5, 0.5, 0.5)).corners
            box_corners_in_image_gt[mask] = points_cam2img(corners_gt, cam2img)

        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        proj_bbox2d_preds = torch.cat([minxy, maxxy], dim=1)

        outputs = (proj_bbox2d_preds, pos_decoded_bbox2d_preds)

        if with_kpts:
            norm_strides = pos_strides * self.regress_ranges[0][1] / \
                self.strides[0]
            kpts_targets = box_corners_in_image_gt - pos_points[..., None, :]
            kpts_targets = kpts_targets.view(
                (*pos_strided_bbox_preds.shape[:-1], 16))
            kpts_targets /= norm_strides

            outputs += (kpts_targets, )

        return outputs