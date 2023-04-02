import torch
import numpy as np
import pdb
from data.cityscapes3d import evalLabels
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFont
import cv2
from pyquaternion import Quaternion
import mmcv
from mmdet3d.core.utils import array_converter
from mmdet3d.core.bbox import CameraInstance3DBoxes

def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (torch.Tensor): The value to be converted.
        offset (float, optional): Offset to set the value range. \
            Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.

        e.g. when offset=0.5, period=2*np.pi, the value range will be [-pi, pi]

    Returns:
        torch.Tensor: Value in the range of \
            [-offset * period, (1-offset) * period]
    """
    return val - torch.floor(val / period + offset) * period

def xywhrst2xyxyrst(boxes_xywhr):
    """Convert a rotated boxes in XYWHrst format to XYXYrst format.
    r: yaw, s: pitch, t: roll

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
    boxes[:, 4:7] = boxes_xywhr[:, 4:7]
    return boxes

def xywhpra2xyxya(boxes_inp): 
    """Convert a rotated boxes in XYWHrst format to XYXYr format.
    p: pitch
    r: roll
    a: yaw

    Args:
        boxes_xywhr (torch.Tensor): Rotated boxes in XYWHR format.

    Returns:
        torch.Tensor: Converted boxes in XYXYR format.
    """
    new_shape = [boxes_inp.shape[0], 5]
    boxes = torch.zeros(new_shape, device=boxes_inp.device, dtype=boxes_inp.dtype)
    half_w = boxes_inp[:, 2] / 2
    half_h = boxes_inp[:, 3] / 2

    boxes[:, 0] = boxes_inp[:, 0] - half_w
    boxes[:, 1] = boxes_inp[:, 1] - half_h
    boxes[:, 2] = boxes_inp[:, 0] + half_w
    boxes[:, 3] = boxes_inp[:, 1] + half_h
    # boxes[:, 4:7] = boxes_xywhr[:, 4:7]
    boxes[:, 4] = boxes_inp[:, 6] # yaw at the last pos among rotations
    return boxes

def bbox_bev(tensor):
    """XYWHyawpitchroll
    Cityscapes "size" is always given in LxWxH
    """

    return tensor[:, [0, 2, 4, 3, 6, 7, 8]]


from .iou3d.iou3d_utils import nms_gpu, nms_normal_gpu
def box3d_multiclass_nms(mlvl_bboxes,
                         mlvl_bboxes_for_nms,
                         mlvl_scores,
                         mlvl_centers2d,
                         score_thr,
                         max_num,
                         cfg,
                         mlvl_dir_scores=None,
                         mlvl_attr_scores=None,
                         mlvl_bboxes2d=None):
    """Multi-class nms for 3D boxes.

    Args:
        mlvl_bboxes (torch.Tensor): Multi-level boxes with shape (N, M).
            M is the dimensions of boxes.
        mlvl_bboxes_for_nms (torch.Tensor): Multi-level boxes with shape
            (N, 5) ([x1, y1, x2, y2, ry]). N is the number of boxes.
        mlvl_scores (torch.Tensor): Multi-level boxes with shape
            (N, C + 1). N is the number of boxes. C is the number of classes.
        score_thr (float): Score thredhold to filter boxes with low
            confidence.
        max_num (int): Maximum number of boxes will be kept.
        cfg (dict): Configuration dict of NMS.
        mlvl_dir_scores (torch.Tensor, optional): Multi-level scores
            of direction classifier. Defaults to None.
        mlvl_attr_scores (torch.Tensor, optional): Multi-level scores
            of attribute classifier. Defaults to None.
        mlvl_bboxes2d (torch.Tensor, optional): Multi-level 2D bounding
            boxes. Defaults to None.

    Returns:
        tuple[torch.Tensor]: Return results after nms, including 3D \
            bounding boxes, scores, labels, direction scores, attribute \
            scores (optional) and 2D bounding boxes (optional).
    """
    # do multi class nms
    # the fg class id range: [0, num_classes-1]
    num_classes = mlvl_scores.shape[1] - 1 # we zero pad one useless dim before, so now we remove it here
    bboxes = []
    scores = []
    labels = []
    centers2d = []
    dir_scores = []
    attr_scores = []
    bboxes2d = []
    for i in range(0, num_classes):
        # get bboxes and scores of this class
        cls_inds = mlvl_scores[:, i] > score_thr
        if not cls_inds.any():
            continue

        _scores = mlvl_scores[cls_inds, i]
        _bboxes_for_nms = mlvl_bboxes_for_nms[cls_inds, :]
        _mlvl_centers2d = mlvl_centers2d[cls_inds, :]

        if cfg.use_rotate_nms:
            nms_func = nms_gpu
        else:
            nms_func = nms_normal_gpu

        selected = nms_func(_bboxes_for_nms, _scores, cfg.nms_thr)
        _mlvl_bboxes = mlvl_bboxes[cls_inds, :]
        bboxes.append(_mlvl_bboxes[selected])
        centers2d.append(_mlvl_centers2d[selected])
        scores.append(_scores[selected])
        cls_label = mlvl_bboxes.new_full((len(selected), ),
                                         i,
                                         dtype=torch.long)
        labels.append(cls_label)

        if mlvl_dir_scores is not None:
            _mlvl_dir_scores = mlvl_dir_scores[cls_inds]
            dir_scores.append(_mlvl_dir_scores[selected])
        if mlvl_attr_scores is not None:
            _mlvl_attr_scores = mlvl_attr_scores[cls_inds]
            attr_scores.append(_mlvl_attr_scores[selected])
        if mlvl_bboxes2d is not None:
            _mlvl_bboxes2d = mlvl_bboxes2d[cls_inds]
            bboxes2d.append(_mlvl_bboxes2d[selected])

    if bboxes:
        bboxes = torch.cat(bboxes, dim=0)
        scores = torch.cat(scores, dim=0)
        labels = torch.cat(labels, dim=0)
        centers2d = torch.cat(centers2d, dim=0)
        if mlvl_dir_scores is not None:
            dir_scores = torch.cat(dir_scores, dim=0)
        if mlvl_attr_scores is not None:
            attr_scores = torch.cat(attr_scores, dim=0)
        if mlvl_bboxes2d is not None:
            bboxes2d = torch.cat(bboxes2d, dim=0)
        if bboxes.shape[0] > max_num:
            _, inds = scores.sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            scores = scores[inds]
            centers2d = centers2d[inds]
            if mlvl_dir_scores is not None:
                dir_scores = dir_scores[inds]
            if mlvl_attr_scores is not None:
                attr_scores = attr_scores[inds]
            if mlvl_bboxes2d is not None:
                bboxes2d = bboxes2d[inds]
    else:
        bboxes = mlvl_scores.new_zeros((0, mlvl_bboxes.size(-1)))
        scores = mlvl_scores.new_zeros((0, ))
        labels = mlvl_scores.new_zeros((0, ), dtype=torch.long)
        centers2d = mlvl_scores.new_zeros((0, mlvl_centers2d.size(-1)))
        if mlvl_dir_scores is not None:
            dir_scores = mlvl_scores.new_zeros((0, ))
        if mlvl_attr_scores is not None:
            attr_scores = mlvl_scores.new_zeros((0, ))
        if mlvl_bboxes2d is not None:
            bboxes2d = mlvl_scores.new_zeros((0, 4))

    results = (bboxes, scores, labels, centers2d)

    if mlvl_dir_scores is not None:
        results = results + (dir_scores, )
    if mlvl_attr_scores is not None:
        results = results + (attr_scores, )
    if mlvl_bboxes2d is not None:
        results = results + (bboxes2d, )

    return results

from cityscapesscripts.helpers.annotation import CsBbox3d
from .box3dImageTransform import (
    Camera, 
    Box3dImageTransform,
    CRS_V,
    CRS_C,
    CRS_S
)

def euler_to_quaternion(yawpitchroll):
    """https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr?rq=1
    """
    new_rot = Rotation.from_euler('ZXY', yawpitchroll)
    # new_rot = Rotation.from_euler('YXZ', yawpitchroll) # under S
    rotation_S = new_rot.as_quat()

    return rotation_S

def get_K_multiplier():
    K_multiplier = np.zeros((3, 3))
    K_multiplier[0][1] = K_multiplier[1][2] = -1
    K_multiplier[2][0] = 1
    return K_multiplier

def quaternion_S2V(quaternion_rot_S, sensor_T_ISO_8855):
    K_multiplier = get_K_multiplier()
    image_T_sensor_quaternion = Quaternion(matrix=K_multiplier)
    quaternion_rot = (
        image_T_sensor_quaternion.inverse *
        quaternion_rot_S *
        image_T_sensor_quaternion
    )
    sensor_T_ISO_8855_quaternion = Quaternion(matrix=np.array(sensor_T_ISO_8855)[:3, :3])
    quaternion_rot = sensor_T_ISO_8855_quaternion.inverse * quaternion_rot
    return quaternion_rot


def bbox2json(bbox, K_matrix, camera_params):
    scores_3d = bbox['img_bbox']['scores_3d']
    boxes_3d = bbox['img_bbox']['boxes_3d']
    labels_3d = bbox['img_bbox']['labels_3d']
    bbox2d = bbox['img_bbox2d']
    sensor_T_ISO_8855 = camera_params["sensor_T_ISO_8855"].cpu().numpy()
    bbox_camera = Camera(fx=camera_params["fx"].cpu().numpy(),
                        fy=camera_params["fy"].cpu().numpy(),
                        u0=camera_params["u0"].cpu().numpy(),
                        v0=camera_params["v0"].cpu().numpy(),
                        sensor_T_ISO_8855=sensor_T_ISO_8855)
    box_no = len(scores_3d)

    out = dict(objects=[])
    for i in range(box_no):
        # already in S
        center_S = boxes_3d[i][:3].numpy().tolist()
        # size is the same under different cooderdinate systems
        size_S = boxes_3d[i][3:6].numpy().tolist() # L, W H
        # convert rotation from yaw-pitch-roll to quaternion under S
        rot = Rotation.from_euler('ZXY', boxes_3d[i][6:9].numpy())
        rotation_S = rot.as_quat()
        # convert from S to V
        box3d_annotation = Box3dImageTransform(camera=bbox_camera)
        box3d_annotation.initialize_box(size=size_S, quaternion=rotation_S, center=center_S, coordinate_system=CRS_S)
        size_V, center_V, rotation_V = box3d_annotation.get_parameters(coordinate_system=CRS_V)
        # 2d amodal bbox
        xmin, ymin, xmax, ymax = box3d_annotation.get_amodal_box_2d()
        w = xmax - xmin
        h = ymax - ymin
        amodal_2dbbox = [xmin, ymin, w, h]
        # 2d modal bbox
        xmin, ymin, xmax, ymax, _score = bbox2d[i].astype(np.float64)
        w = xmax - xmin
        h = ymax - ymin
        modal_2dbbox = [xmin, ymin, w, h]

        # labels
        label = labels_3d[i].item()
        label = evalLabels[label]
        score = scores_3d[i].item()   # used for evaluating under different threshold
        out['objects'].append(
            {'2d': {
                'amodal': amodal_2dbbox,
                'modal': modal_2dbbox
            },
            '3d': {
                "center": center_V.tolist(),
                "dimensions": size_V.tolist(),
                "rotation": rotation_V.q.tolist(),
            },
            "label": label,
            "score": score
            })
    return out

corner_mm2cs = ['BLT', 'FLT', 'FLB', 'BLB', 'BRT', 'FRT', 'FRB', 'BRB']
assert len(set(corner_mm2cs)) == 8
# corner_cs2mm_dict = {corner_mm2cs[ind]: ind for ind in len(corner_mm2cs)}

def plot_rect3d_on_img(img,
                       num_rects,
                       rect_corners,
                       color=(0, 255, 0),
                       thickness=1):
    """Plot the boundary lines of 3D rectangular on 2D images.

        Convert the boxes to  in clockwise order, in the form of
        (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)

        .. code-block:: none

                         front z
                              /
                             /
               (x0, y0, z1) + -----------  + (x1, y0, z1)
                           /|            / |
                          / |           /  |
            (x0, y0, z0) + ----------- +   + (x1, y1, z1)
                         |  /      .   |  /
                         | / origin    | /
            (x0, y1, z0) + ----------- + -------> x right
                         |             (x1, y1, z0)
                         |
                         v
                    down y

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))

    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int)
        for start, end in line_indices:
            cv2.line(img, (corners[start, 0], corners[start, 1]),
                        (corners[end, 0], corners[end, 1]), color, thickness,
                        cv2.LINE_AA)

    return img.astype(np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
def bbox2fig(p, inp, bbox, K_matrix, camera_params, gt_labels=None):
    scores_3d = bbox['img_bbox']['scores_3d']
    boxes_3d = bbox['img_bbox']['boxes_3d']
    labels_3d = bbox['img_bbox']['labels_3d']
    centersI = bbox['img_bbox']['centers2d']

    box_no = len(scores_3d)

    image = inp.numpy().transpose(1,2,0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image*std + mean
    image = image * 255

    image = image.astype(np.uint8) 
    image = Image.fromarray(image)
    width_height = (p.IMAGE_ORI_SIZE[1], p.IMAGE_ORI_SIZE[0])
    image = image.resize(width_height, Image.BILINEAR)
    draw = ImageDraw.Draw(image)
    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)

    if False:
        draw.text((10, 10), str(box_no), font=fnt, fill=(0,0,255))

    # gt bbox number
    if gt_labels != None:
        gt_class, gt_center_I, gt_center_S, gt_size_S, gt_rotation_S = gt_labels
    else:
        gt_class = None

    if gt_class != None:
        gt_box_no = gt_center_S.shape[0]
        if False:
            draw.text((10, 60), str(gt_box_no), font=fnt, fill=(0,255,0))

    all_boxes = []

    bbox_camera = Camera(fx=camera_params["fx"].cpu().numpy(),
                        fy=camera_params["fy"].cpu().numpy(),
                        u0=camera_params["u0"].cpu().numpy(),
                        v0=camera_params["v0"].cpu().numpy(),
                        sensor_T_ISO_8855=camera_params["sensor_T_ISO_8855"].cpu().numpy())
    for i in range(box_no):
        # vis_center
        center_2d = centersI[i, :2]
        if False:
            draw.ellipse([tuple(center_2d-3), tuple(center_2d+3)], fill=(0,0,255))
            draw.text((center_2d[0]-3, center_2d[1]-3), evalLabels[labels_3d[i].data], font=fnt, fill=(0,0,255))

        # vis bbox
        center_S = boxes_3d[i][:3].numpy().tolist()
        # size is the same under different cooderdinate systems
        size_S = boxes_3d[i][3:6].numpy().tolist() # L, W H
        # convert rotation from yaw-pitch-roll to quaternion under S
        rot = Rotation.from_euler('ZXY', boxes_3d[i][6:9].numpy())
        rotation_S = rot.as_quat()
        # convert from S to V
        box3d_annotation = Box3dImageTransform(camera=bbox_camera)
        box3d_annotation.initialize_box(size=size_S, quaternion=rotation_S, center=center_S, coordinate_system=CRS_S)

        # 2d amodal bbox
        if False:
            xmin, ymin, xmax, ymax = box3d_annotation.get_amodal_box_2d()
            draw.rectangle([xmin, ymin, xmax, ymax])

        # 3d vertices to 2d plane
        box_vertices_I = box3d_annotation.get_vertices_2d()
        # loc is encoded with a 3-char code
        #   0: B/F: Back or Front
        #   1: L/R: Left or Right
        #   2: B/T: Bottom or Top
        # BLT -> Back left top of the object
        if False:
            for corner, coor in box_vertices_I.items():
                draw.ellipse([tuple(coor-3), tuple(coor+3)], fill=(0,255,0))
        corners = []
        for k in corner_mm2cs:
            corners.append(box_vertices_I[k])
        corners = np.array(corners).astype(np.int)    
        all_boxes.append(corners)

    image = np.array(image)
    if box_no > 0:
        all_boxes = np.array(all_boxes)
        try:
            # plot_rect3d_on_img(image, all_boxes.shape[0], all_boxes, color=(0, 0, 255), thickness=1)
            plot_rect3d_on_img(image, all_boxes.shape[0], all_boxes, color=(255, 139, 255), thickness=2)
        except:
            print('size_S')
            print(size_S)

    # Visualize the gt boxes
    # get gt labels if any
    if gt_class != None:
        gt_box_no = gt_center_S.shape[0]
        if False:
            draw.text((10, 20), str(gt_box_no), font=fnt, fill=(0,255,0))
        all_boxes = []
        for i in range(gt_box_no):
            cls, center_I, center_S, size_S, rotation_S = gt_class[i], gt_center_I[i].tolist(), gt_center_S[i].tolist(), gt_size_S[i].tolist(), euler_to_quaternion(gt_rotation_S[i])
            box3d_annotation = Box3dImageTransform(camera=bbox_camera)
            box3d_annotation.initialize_box(size=size_S, quaternion=rotation_S, center=center_S, coordinate_system=CRS_S)
            # plot gt bbox
            box_vertices_I = box3d_annotation.get_vertices_2d()
            corners = []
            for k in corner_mm2cs:
                corners.append(box_vertices_I[k])
            corners = np.array(corners).astype(np.int)    
            all_boxes.append(corners)

            # plot center
            if False:
                image = cv2.circle(image, (int(center_I[0]), int(center_I[1])), radius=2, color=(0, 255, 0), thickness=2)
                image = cv2.putText(image, evalLabels[cls.data], (int(center_I[0]), int(center_I[1])), font, fontScale,
                            (0,255,0), 2, cv2.LINE_AA)

        if gt_box_no > 0:
            all_boxes = np.array(all_boxes)
            plot_rect3d_on_img(image, all_boxes.shape[0], all_boxes, color=(0, 255, 0))

    return image

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.

    Returns:
        Tensor: Boxes with shape (N, 4) or (B, N, 4)
    """

    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    if max_shape is not None:
        if bboxes.dim() == 2 and not torch.onnx.is_in_onnx_export():
            # speed up
            bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
            bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
            return bboxes

        # clip bboxes with dynamic `min` and `max` for onnx
        if torch.onnx.is_in_onnx_export():
            from mmdet.core.export import dynamic_clip_for_onnx
            x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, max_shape)
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
            return bboxes
        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(x1)
        if max_shape.ndim == 2:
            assert bboxes.ndim == 3
            assert max_shape.size(0) == bboxes.size(0)

        min_xy = x1.new_tensor(0)
        max_xy = torch.cat([max_shape, max_shape],
                           dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes

def bbox3d2result(bboxes, scores, labels, centers2d):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
        labels (torch.Tensor): Labels with shape of (n, ).
        scores (torch.Tensor): Scores with shape of (n, ).
        attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
    """
    result_dict = dict(
        boxes_3d=bboxes.cpu(),
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu(),
        centers2d=centers2d.cpu())

    return result_dict

def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float64) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
        return bboxes

@array_converter(apply_to=('points_3d', 'proj_mat'))
def points_cam2img(points_3d, proj_mat, with_depth=False):
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

    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        point_2d_res = torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)

    return point_2d_res

@array_converter(apply_to=('points', 'cam2img'))
def points_img2cam(points, cam2img):
    """Project points in image coordinates to camera coordinates.

    Args:
        points (torch.Tensor): 2.5D points in 2D images, [N, 3],
            3 corresponds with x, y in the image and depth.
        cam2img (torch.Tensor): Camera intrinsic matrix. The shape can be
            [3, 3], [3, 4] or [4, 4].

    Returns:
        torch.Tensor: points in 3D space. [N, 3],
            3 corresponds with x, y, z in 3D space.
    """
    assert cam2img.shape[0] <= 4
    assert cam2img.shape[1] <= 4
    assert points.shape[1] == 3

    xys = points[:, :2]
    depths = points[:, 2].view(-1, 1)
    unnormed_xys = torch.cat([xys * depths, depths], dim=1)

    pad_cam2img = torch.eye(4, dtype=xys.dtype, device=xys.device)
    pad_cam2img[:cam2img.shape[0], :cam2img.shape[1]] = cam2img
    inv_pad_cam2img = torch.inverse(pad_cam2img).transpose(0, 1)

    # Do operation in homogeneous coordinates.
    num_points = unnormed_xys.shape[0]
    homo_xys = torch.cat([unnormed_xys, xys.new_ones((num_points, 1))], dim=1)
    points3D = torch.mm(homo_xys, inv_pad_cam2img)[:, :3]

    return points3D

def decode_yaw(bbox, centers2d, dir_cls, dir_offset, cam2img):
    """Decode yaw angle and change it from local to global.i.

    Args:
        bbox (torch.Tensor): Bounding box predictions in shape
            [N, C] with yaws to be decoded.
        centers2d (torch.Tensor): Projected 3D-center on the image planes
            corresponding to the box predictions.
        dir_cls (torch.Tensor): Predicted direction classes.
        dir_offset (float): Direction offset before dividing all the
            directions into several classes.
        cam2img (torch.Tensor): Camera intrinsic matrix in shape [4, 4].

    Returns:
        torch.Tensor: Bounding boxes with decoded yaws.
    """
    if bbox.shape[0] > 0:
        for rot_i, rot in enumerate(range(6, 9)): # all euler angles
            dir_rot = limit_period(bbox[..., rot] - dir_offset, 0,
                                np.pi) # this is becoz fcos3d uses direction classification so it only needs to predict angle in range (0, pi)
            bbox[..., rot] = \
                dir_rot + dir_offset + np.pi * dir_cls[..., rot_i].to(bbox.dtype)

    # bbox[:, 6] = torch.atan2(centers2d[:, 0] - cam2img[0, 2],
    #                             cam2img[0, 0]) + bbox[:, 6]
    return bbox