# By Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch.nn as nn
import torch.nn.functional as F

INTERPOLATE_MODE = 'bilinear'

class TaskPrompterWrapper(nn.Module):
    def __init__(self, p, backbone, heads):
        super(TaskPrompterWrapper, self).__init__()
        self.tasks = p.TASKS.NAMES

        self.backbone = backbone
        self.heads = heads 

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}

        # TaskPrompter
        task_features, info = self.backbone(x) 

        # Generate predictions
        out = {}
        for t in self.tasks:
            _task_fea = task_features[t]
            if t != '3ddet':
                out[t] = F.interpolate(self.heads[t](_task_fea), img_size, mode=INTERPOLATE_MODE)
            else:
                out[t] = self.heads[t](_task_fea)
            
        return out
