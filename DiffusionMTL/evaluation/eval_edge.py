#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import glob
import json
import torch
import numpy as np
from utils.utils import mkdir_if_missing
from losses.loss_functions import BalancedBinaryCrossEntropyLoss
from utils.mypath import PROJECT_ROOT_DIR

class EdgeMeter(object):
    def __init__(self, p):
        self.loss = 0
        self.n = 0
        self.loss_function = BalancedBinaryCrossEntropyLoss(pos_weight=p['edge_w'], ignore_index=p.ignore_index)
        
    @torch.no_grad()
    def update(self, pred, gt):
        gt = gt.squeeze()
        pred = pred.float().squeeze() / 255.
        loss = self.loss_function(pred, gt).item()
        numel = gt.numel()
        self.n += numel
        self.loss += numel * loss

    def reset(self):
        self.loss = 0
        self.n = 0

    def get_score(self, verbose=True):
        eval_dict = {'loss': self.loss / self.n}

        if verbose:
            print('\n Edge Detection Evaluation')
            print('Edge Detection Loss %.3f' %(eval_dict['loss']))

        return eval_dict