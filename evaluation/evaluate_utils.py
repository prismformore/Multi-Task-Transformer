# This code is referenced from MTI-Net
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import cv2
import imageio
import numpy as np
import json
import torch
import scipy.io as sio
from utils.utils import get_output
import pdb


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
        return DepthMeter(ignore_index=p.ignore_index) 

    elif task == 'edge': # just for reference
        from evaluation.eval_edge import EdgeMeter
        return EdgeMeter(pos_weight=p['edge_w'], ignore_index=p.ignore_index)

    else:
        raise NotImplementedError

@torch.no_grad()
def save_model_pred_for_one_task(p, sample, output, save_dirs, task=None, epoch=None):
    """ Save model predictions for one task"""

    inputs, meta = sample['image'].cuda(non_blocking=True), sample['meta']
    output_task = get_output(output[task], task)

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
                # deprecated by python
                # height_location = [delta_height // 2,
                #                    (delta_height // 2) + im_height]
                # width_location = [delta_width // 2,
                #                   (delta_width // 2) + im_width]
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
