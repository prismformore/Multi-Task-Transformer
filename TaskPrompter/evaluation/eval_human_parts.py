# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import warnings
import cv2
import glob
import json
import os.path
import numpy as np
import torch
from PIL import Image

PART_CATEGORY_NAMES = ['background', 'head', 'torso', 'uarm', 'larm', 'uleg', 'lleg']

class HumanPartsMeter(object):
    def __init__(self, database, ignore_idx=255):
        assert(database == 'PASCALContext')
        self.database = database
        self.cat_names = PART_CATEGORY_NAMES
        self.n_parts = 6
        self.tp = [0] * (self.n_parts + 1)
        self.fp = [0] * (self.n_parts + 1)
        self.fn = [0] * (self.n_parts + 1)

        self.ignore_idx = ignore_idx

    @torch.no_grad() 
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()
        valid = (gt != self.ignore_idx)
        
        for i_part in range(self.n_parts + 1):
            tmp_gt = (gt == i_part)
            tmp_pred = (pred == i_part)
            self.tp[i_part] += torch.sum(tmp_gt & tmp_pred & (valid)).item()
            self.fp[i_part] += torch.sum(~tmp_gt & tmp_pred & (valid)).item()
            self.fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & (valid)).item()

    def reset(self):
        self.tp = [0] * (self.n_parts + 1)
        self.fp = [0] * (self.n_parts + 1)
        self.fn = [0] * (self.n_parts + 1)
 
    def get_score(self, verbose=True):
        jac = [0] * (self.n_parts + 1)
        for i_part in range(0, self.n_parts + 1):
            jac[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)

        eval_result = dict()
        # eval_result['jaccards_all_categs'] = jac
        eval_result['mIoU'] = np.mean(jac) * 100
        
        print('\nHuman Parts mIoU: {0:.4f}\n'.format(eval_result['mIoU']))
        class_IoU = jac
        for i in range(len(class_IoU)):
            spaces = ''
            for j in range(0, 15 - len(self.cat_names[i])):
                spaces += ' '
            print('{0:s}{1:s}{2:.4f}'.format(self.cat_names[i], spaces, 100 * class_IoU[i]))

        return eval_result
