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

        # 3d det
        if '3ddet' in self.tasks:
            self.do_3d_det = True
            self.detmodel = p.detmodel
        else:
            self.do_3d_det = False

    
    def forward(self, pred, gt, tasks):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in tasks if task != '3ddet'}

        # det loss
        if '3ddet' in tasks:
            det_losses, det_loss_sum = self.detmodel.loss(pred['3ddet'], gt) 
            
            out['3ddet'] = det_loss_sum
            out = {**out, **det_losses}

        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in tasks]))

        return out
