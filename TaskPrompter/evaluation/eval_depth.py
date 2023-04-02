# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

from shutil import ignore_patterns
import warnings
import cv2
import os.path
import numpy as np
import glob
import torch
import json
import scipy.io as sio

class DepthMeter(object):
    def __init__(self, max_depth=None, min_depth=None):
        self.total_rmses = 0.0
        self.total_log_rmses = 0.0
        self.n_valid = 0.0
        self.max_depth = max_depth
        self.min_depth = min_depth

        self.abs_rel = 0.0
        self.sq_rel = 0.0

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()
        
        # Determine valid mask
        # mask = (gt != self.ignore_index).bool()
        mask = torch.logical_and(gt < self.max_depth, gt > self.min_depth)
        self.n_valid += mask.float().sum().item() # Valid pixels per image
        
        # Only positive depth values are possible
        # pred = torch.clamp(pred, min=1e-9)
        gt[gt <=0 ] = 1e-9
        pred[pred <= 0] = 1e-9

        # Per pixel rmse and log-rmse.
        log_rmse_tmp = torch.pow(torch.log(gt[mask]) - torch.log(pred[mask]), 2)
        self.total_log_rmses += log_rmse_tmp.sum().item()

        rmse_tmp = torch.pow(gt[mask] - pred[mask], 2)
        self.total_rmses += rmse_tmp.sum().item()

        # abs rel
        self.abs_rel += (torch.abs(gt[mask] - pred[mask]) / gt[mask]).sum().item()
        # sq_rel
        self.sq_rel += (((gt[mask] - pred[mask]) ** 2) / gt[mask]).sum().item()

    def get_score(self, verbose=True):
        eval_result = dict()
        eval_result['rmse'] = np.sqrt(self.total_rmses / self.n_valid)
        eval_result['log_rmse'] = np.sqrt(self.total_log_rmses / self.n_valid)
        eval_result['abs_rel'] = self.abs_rel / self.n_valid
        eval_result['sq_rel'] = self.sq_rel / self.n_valid

        if verbose:
            print('Results for depth prediction')
            for x in eval_result:
                spaces = ''
                for j in range(0, 15 - len(x)):
                    spaces += ' '
                print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_result[x]))

        return eval_result

class DepthMeter_legacy(object):
    def __init__(self, ignore_index=255):
        self.total_rmses = 0.0
        self.total_log_rmses = 0.0
        self.n_valid = 0.0
        self.ignore_index = ignore_index

        self.abs_rel = 0.0
        self.sq_rel = 0.0

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()
        
        # Determine valid mask
        mask = (gt != self.ignore_index).bool()
        self.n_valid += mask.float().sum().item() # Valid pixels per image
        
        # Only positive depth values are possible
        # pred = torch.clamp(pred, min=1e-9)
        gt[gt <=0 ] = 1e-9
        pred[pred <= 0] = 1e-9

        # Per pixel rmse and log-rmse.
        log_rmse_tmp = torch.pow(torch.log(gt[mask]) - torch.log(pred[mask]), 2)
        self.total_log_rmses += log_rmse_tmp.sum().item()

        rmse_tmp = torch.pow(gt[mask] - pred[mask], 2)
        self.total_rmses += rmse_tmp.sum().item()

        # abs rel
        self.abs_rel += (torch.abs(gt[mask] - pred[mask]) / gt[mask]).sum().item()
        # sq_rel
        self.sq_rel += (((gt[mask] - pred[mask]) ** 2) / gt[mask]).sum().item()

    def reset(self):
        self.rmses = []
        self.log_rmses = []
        
    def get_score(self, verbose=True):
        eval_result = dict()
        eval_result['rmse'] = np.sqrt(self.total_rmses / self.n_valid)
        eval_result['log_rmse'] = np.sqrt(self.total_log_rmses / self.n_valid)
        eval_result['abs_rel'] = self.abs_rel / self.n_valid
        eval_result['sq_rel'] = self.sq_rel / self.n_valid

        if verbose:
            print('Results for depth prediction')
            for x in eval_result:
                spaces = ''
                for j in range(0, 15 - len(x)):
                    spaces += ' '
                print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_result[x]))

        return eval_result
