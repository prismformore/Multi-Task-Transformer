import pdb
import torch
import numpy as np
import logging
import os

from data.cityscapes3d import evalLabels
from cityscapesscripts.evaluation.objectDetectionHelpers import (
    EvaluationParameters,
    getFiles,
    calcIouMatrix,
    calcOverlapMatrix
)
from cityscapesscripts.evaluation.objectDetectionHelpers import (
    MATCHING_MODAL,
    MATCHING_AMODAL
)
from .evalObjectDetection3d import evaluate3dObjectDetection

def eval_3ddet(p, save_dir):
    # Follow official setting and eval on val set
    # set as default evalution params
    eval_params = EvaluationParameters(
        evalLabels,
        min_iou_to_match=0.7,
        max_depth=100,
        step_size=5,
        matching_method=MATCHING_MODAL,
        cw=-1.
    )

    print("===================================")
    print("=== Start CS val set evaluation ===")
    print("===================================")
    results = evaluate3dObjectDetection(
        gt_folder=os.path.join(p.db_paths['Cityscapes3D'], 'gtBbox3d', 'val'),
        pred_folder=os.path.join(save_dir, '3ddet'),
        result_folder=save_dir,
        eval_params=eval_params,
        plot=False,
    )
    print("===================================")
    print("=== Finish CS val set evaluation ===")
    print("===================================")
    return results

class DetMeter(object):
    def __init__(self, p):
        database = p['train_db_name']
        if database == 'Cityscapes3D':
            n_classes = 9
            det_cls_labels = ['car', 'truck', 'bus', 'train', \
                              'motorcycle', 'bicycle', 'dynamic', \
                               'trailer', 'tunnel', 'caravan'] 
                               # totally 9 classes, caravan is not in train set.

            has_bg = False
        
        else:
            raise NotImplementedError
        
        self.n_classes = n_classes + int(has_bg)
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes

        self.ignore_idx = 250

    @torch.no_grad()
    def update(self, pred, label):
        pdb.set_trace()
        

    def reset(self):
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes
            
    def get_score(self, verbose=True):
        jac = [0] * self.n_classes
        for i_part in range(self.n_classes):
            jac[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)

        eval_result = dict()
        eval_result['jaccards_all_categs'] = jac
        eval_result['mIoU'] = np.mean(jac)


        if verbose:
            print('\nSemantic Segmentation mIoU: {0:.4f}\n'.format(100 * eval_result['mIoU']))
            class_IoU = eval_result['jaccards_all_categs']
            for i in range(len(class_IoU)):
                spaces = ''
                for j in range(0, 20 - len(self.cat_names[i])):
                    spaces += ' '
                print('{0:s}{1:s}{2:.4f}'.format(self.cat_names[i], spaces, 100 * class_IoU[i]))

        return eval_result