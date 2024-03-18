
import os
import cv2
import imageio
import numpy as np
import json
import torch
import scipy.io as sio
from utils.utils import get_output, mkdir_if_missing
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
            if t == '3ddet':
                self.meters[t].update(pred[t], gt)
            else:
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
        return NormalsMeter(ignore_index=p.ignore_index) #NormalsMeter()

    elif task == 'sal':
        from evaluation.eval_sal import SaliencyMeter
        return SaliencyMeter(ignore_index=255, threshold_step=0.05, beta_squared=0.3)

    elif task == 'depth':
        from evaluation.eval_depth import DepthMeter
        return DepthMeter(ignore_index=p.ignore_index) 

    elif task == 'edge': # Single task performance meter uses the loss (True evaluation is based on seism evaluation)
        from evaluation.eval_edge import EdgeMeter
        return EdgeMeter(p=p)

    elif task == '3ddet':
        raise NotImplementedError # TODO
        from detection.det_eval import DetMeter
        return DetMeter(p)

    else:
        raise NotImplementedError


@torch.no_grad()
def save_model_pred_for_one_task(p, sample, output, save_dirs, task=None, epoch=None):
    """ Save model predictions for one task"""

    inputs, meta = sample['image'], sample['meta']
    img_size = (inputs.size(2), inputs.size(3))

    if task == 'semseg':
        if not p.semseg_save_train_class and p.train_db_name == 'Cityscapes':
            output_task = get_output(output[task], task, semseg_save_train_class=False).cpu().data.numpy()
        else:
            output_task = get_output(output[task], task).cpu().data.numpy()
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
                height_location = [delta_height // 2,
                                   (delta_height // 2) + im_height]
                width_location = [delta_width // 2,
                                  (delta_width // 2) + im_width]
                pred = pred[height_location[0]:height_location[1],
                            width_location[0]:width_location[1]]
        assert pred.shape[:2] == (im_height, im_width)
        if pred.ndim == 3:
            raise
            # pred = pred.permute(1, 2, 0) # pred.transpose(1, 2, 0) this is a very old grammar?
        result = pred.cpu().numpy()
        # result = cv2.resize(output_task[jj], dsize=(int(meta['img_size'][jj][1]), int(meta['img_size'][jj][0])), interpolation=p.TASKS.INFER_FLAGVALS[task])
        if task == 'depth':
            sio.savemat(os.path.join(save_dirs[task], fname + '.mat'), {'depth': result})
        else:
            imageio.imwrite(os.path.join(save_dirs[task], fname + '.png'), result.astype(np.uint8))

