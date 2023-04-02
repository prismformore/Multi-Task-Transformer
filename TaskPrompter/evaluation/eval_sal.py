# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import torch
from torch import nn

class SaliencyMeter(object):
    def __init__(self, ignore_index=255, threshold_step=None, beta_squared=1):
        self.ignore_index = ignore_index
        self.beta_squared = beta_squared
        self.thresholds = torch.arange(threshold_step, 1, threshold_step)
        self.true_positives = torch.zeros(len(self.thresholds))
        self.predicted_positives = torch.zeros(len(self.thresholds))
        self.actual_positives = torch.zeros(len(self.thresholds))

    @torch.no_grad()
    def update(self, preds, target):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model [B, H, W]
            target: Ground truth values
        """
        preds = preds.float() / 255.

        if target.shape[1] == 1:
            target = target.squeeze(1)

        assert preds.shape == target.shape

        if len(preds.shape) == len(target.shape) + 1:
            assert preds.shape[1] == 2
            # two class probabilites
            preds = nn.functional.softmax(preds, dim=1)[:, 1, :, :]
        else:
            # squash logits into probabilities
            preds = torch.sigmoid(preds)

        if not len(preds.shape) == len(target.shape):
            raise ValueError("preds and target must have same number of dimensions, or preds one more")

        valid_mask = (target != self.ignore_index)

        for idx, thresh in enumerate(self.thresholds):
            # threshold probablities
            f_preds = (preds >= thresh).long()
            f_target = target.long()

            f_preds = torch.masked_select(f_preds, valid_mask)
            f_target = torch.masked_select(f_target, valid_mask)

            self.true_positives[idx] += torch.sum(f_preds * f_target).cpu()
            self.predicted_positives[idx] += torch.sum(f_preds).cpu()
            self.actual_positives[idx] += torch.sum(f_target).cpu()


    def get_score(self, verbose=False):    
        """
        Computes F-scores over state and returns the max.
        """
        precision = self.true_positives.float() / self.predicted_positives
        recall = self.true_positives.float() / self.actual_positives

        num = (1 + self.beta_squared) * precision * recall
        denom = self.beta_squared * precision + recall

        # For the rest we need to take care of instances where the denom can be 0
        # for some classes which will produce nans for that class
        fscore = num / denom
        fscore[fscore != fscore] = 0

        eval_result = {'maxF': fscore.max().item() * 100}
        return eval_result
