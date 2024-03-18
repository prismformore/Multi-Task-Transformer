import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import pdb

class SemiSupervisedMultiTaskLoss(nn.Module):
    def __init__(self, p, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(SemiSupervisedMultiTaskLoss, self).__init__()
        assert(set(tasks) == set(loss_ft.keys()))
        assert(set(tasks) == set(loss_weights.keys()))
        self.p = p
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

        self.mse =  torch.nn.MSELoss()

        ##### get task_mask ####
        ssl_type = p.ssl_type
        if p.train_db_name == 'PASCALContext':
            labelroot = './data/ssl_mapping/pascal/new_'
            if ssl_type == 'onelabel':
                self.labels_weights = torch.load('{}onelabel.pth'.format(labelroot))
            elif ssl_type == 'randomlabels':
                self.labels_weights = torch.load('{}randomlabels.pth'.format(labelroot))

        self.inter_loss_weight = 0.5
    
    def forward(self, pred, gt, tasks):
        inter_pred = pred['inter_preds']
        pred = pred['preds']
        out = {task: 0 for task in tasks}
        out.update({'inter_'+task: 0 for task in tasks})
        out['total'] = 0

        # for paritial labels available
        local_bs = gt['semseg'].shape[0]

        image_index = gt['meta']['img_name']

        for idx in range(local_bs):
            w = self.labels_weights[image_index[idx]] #.clone().float().cuda()
            label_count = 0
            sample_loss = 0
            for t_idx, task in enumerate(tasks):
                if w[t_idx] == 1:
                    label_count += 1
                    cur_loss = self.loss_ft[task](pred[idx][task], gt[task][idx:idx+1]) / local_bs 
                    out[task] += cur_loss.detach()
                    sample_loss += cur_loss * self.loss_weights[task]

                    # intermediate supervision
                    cur_loss = self.loss_ft[task](inter_pred[task][idx:idx+1], gt[task][idx:idx+1]) / local_bs 
                    out['inter_'+task] += cur_loss.detach()
                    sample_loss += cur_loss * self.loss_weights[task] * self.inter_loss_weight

            # balance loss scale
            if label_count == 0:
                pass
            else:
                out['total'] += sample_loss * len(tasks) / label_count

        return out
