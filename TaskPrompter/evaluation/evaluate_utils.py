# Hanrong Ye

import os
import cv2
import imageio
import numpy as np
import json
import torch
import scipy.io as sio
from utils.utils import get_output
# from detection_toolbox.det_tools import bbox2json, bbox2fig

class PerformanceMeter(object):
    """ A general performance meter which shows performance across one or more tasks """
    def __init__(self, p, tasks):
        self.database = p['train_db_name']
        self.tasks = tasks
        self.meters = {t: get_single_task_meter(p, self.database, t) for t in self.tasks}

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()

    def update(self, pred, gt):
        for t in self.tasks:
            self.meters[t].update(pred[t], gt[t])

    def get_score(self, verbose=True):
        eval_dict = {}
        for t in self.tasks:
            eval_dict[t] = self.meters[t].get_score(verbose)

        return eval_dict

def get_single_task_meter(p, database, task):
    """ Retrieve a meter to measure the single-task performance """

    # ignore index based on transforms.AddIgnoreRegions
    if task == 'semseg':
        from evaluation.eval_semseg import SemsegMeter
        return SemsegMeter(database, ignore_idx=p.ignore_index)

    elif task == 'human_parts':
        from evaluation.eval_human_parts import HumanPartsMeter
        return HumanPartsMeter(database, ignore_idx=p.ignore_index)

    elif task == 'normals':
        from evaluation.eval_normals import NormalsMeter
        return NormalsMeter(ignore_index=p.ignore_index) 

    elif task == 'sal':
        from evaluation.eval_sal import  SaliencyMeter
        return SaliencyMeter(ignore_index=p.ignore_index, threshold_step=0.05, beta_squared=0.3)

    elif task == 'depth':
        from evaluation.eval_depth import DepthMeter
        # Set effective depth evaluation range. Refer to:
        # https://github.com/sjsu-smart-lab/Self-supervised-Monocular-Trained-Depth-Estimation-using-Self-attention-and-Discrete-Disparity-Volum/blob/3c6f46ab03cfd424b677dfeb0c4a45d6269415a9/evaluate_city_depth.py#L55
        return DepthMeter(max_depth=p.TASKS.depth_max, min_depth=p.TASKS.depth_min) 

    elif task == 'edge': # just for reference
        from evaluation.eval_edge import EdgeMeter
        return EdgeMeter(pos_weight=p['edge_w'], ignore_index=p.ignore_index)

    else:
        raise NotImplementedError

@torch.no_grad()
def save_model_pred_for_one_task(p, batch_idx, sample, output, save_dirs, task=None, epoch=None):
    """ Save model predictions for one task"""

    inputs, meta = sample['image'].cuda(non_blocking=True), sample['meta']

    if task == 'semseg':
        if not p.semseg_save_train_class and p.train_db_name == 'Cityscapes3D':
            output_task = get_output(output[task], task, semseg_save_train_class=False).cpu().data.numpy()
        else:
            output_task = get_output(output[task], task).cpu().data.numpy()

    elif task == '3ddet': # save only the first iteraction in an epoch for examing the performance
        from detection_toolbox.det_tools import bbox2json, bbox2fig
        det_res_list = get_output(output[task], task, p=p, label=sample)
        bs = int(inputs.size()[0])
        K_matrixes = sample['meta']['K_matrix'].cpu().numpy()
        cam_params = [{k: v[sa] for k, v in sample['bbox_camera_params'].items()} for sa in range(bs)]

        if batch_idx == 0:
            # get gt labels 
            gt_center_I = []
            gt_center_S = []
            gt_size_S = []
            gt_rotation_S = []
            gt_class = []
            for _i in range(bs):
                if type(sample['det_labels'][_i]) == dict:
                    gt_center_I.append(sample['det_labels'][_i]['center_I'].cpu().numpy())
                    gt_center_S.append(sample['det_labels'][_i]['center_S'].cpu().numpy())
                    gt_size_S.append(sample['det_labels'][_i]['size_S'].cpu().numpy())
                    gt_rotation_S.append(sample['det_labels'][_i]['rotation_S'].cpu().numpy())
                    gt_class.append(sample['det_labels'][_i]['label'])
                else:
                    gt_center_I.append(None)
                    gt_center_S.append(None)
                    gt_size_S.append(None)
                    gt_rotation_S.append(None)
                    gt_class.append(None)

        for jj in range(bs):
            fname = meta['img_name'][jj]
            vis_fname = 'it' + str(epoch) + '_' + meta['img_name'][jj]
            # save bbox predictions in cityscapes evaluation format
            json_dict = bbox2json(det_res_list[jj], K_matrixes[jj], cam_params[jj])
            out_path = os.path.join(save_dirs[task], fname + '.json')
            with open(out_path, 'w') as outfile:
                json.dump(json_dict, outfile)
            if True and batch_idx ==0:
                # visualization, but it takes time so we only use it in infer mode
                box_no = len(det_res_list[jj]['img_bbox']['scores_3d'])
                if box_no > 0:
                    gt_labels = [gt_class[jj], gt_center_I[jj], gt_center_S[jj], gt_size_S[jj], gt_rotation_S[jj]]
                    vis_fig = bbox2fig(p, inputs[jj].cpu(), det_res_list[jj], K_matrixes[jj], cam_params[jj], gt_labels)
                    imageio.imwrite(os.path.join(save_dirs[task], vis_fname + '_' + str(box_no) + '.png'), vis_fig.astype(np.uint8))

        return
    else:
        output_task = get_output(output[task], task)#.cpu().data.numpy()

    for jj in range(int(inputs.size()[0])):
        if len(sample[task][jj].unique()) == 1 and sample[task][jj].unique() == p.ignore_index:
            continue
        fname = meta['img_name'][jj]

        im_height = meta['img_size'][jj][0]
        im_width = meta['img_size'][jj][1]
        pred = output_task[jj] # (H, W) or (H, W, C)
        # if we used padding on the input, we crop the prediction accordingly
        if (im_height, im_width) != pred.shape[:2]:
            delta_height = max(pred.shape[0] - im_height, 0)
            delta_width = max(pred.shape[1] - im_width, 0)
            if delta_height > 0 or delta_width > 0:
                height_begin = torch.div(delta_height, 2, rounding_mode="trunc")
                height_location = [height_begin, height_begin + im_height]
                width_begin =torch.div(delta_width, 2, rounding_mode="trunc")
                width_location = [width_begin, width_begin + im_width]
                pred = pred[height_location[0]:height_location[1],
                            width_location[0]:width_location[1]]
        assert pred.shape[:2] == (im_height, im_width)
        if pred.ndim == 3:
            raise
        result = pred.cpu().numpy()
        if task == 'depth':
            sio.savemat(os.path.join(save_dirs[task], fname + '.mat'), {'depth': result})
        else:
            imageio.imwrite(os.path.join(save_dirs[task], fname + '.png'), result.astype(np.uint8))
