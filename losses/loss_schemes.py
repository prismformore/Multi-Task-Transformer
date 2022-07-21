#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self, p, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MultiTaskLoss, self).__init__()
        assert(set(tasks) == set(loss_ft.keys()))
        assert(set(tasks) == set(loss_weights.keys()))
        self.p = p
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    
    def forward(self, pred, gt, tasks):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in tasks}

        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in tasks]))

        if self.p.intermediate_supervision:
            inter_preds = pred['inter_preds']
            losses_inter = {t: self.loss_ft[t](inter_preds[t], gt[t]) for t in self.tasks}
            for k, v in losses_inter.items():
                out['inter_%s' %(k)] = v
                out['total'] += self.loss_weights[k] * v #* 0.5

        return out
