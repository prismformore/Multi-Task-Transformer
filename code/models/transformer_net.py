# By Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformers.transformer_decoder import TransformerDecoder 
import pdb

from easydict import EasyDict as edict
INTERPOLATE_MODE = 'bilinear'

class TransformerNet(nn.Module):
    def __init__(self, p, backbone, backbone_channels, heads):
        super(TransformerNet, self).__init__()
        self.tasks = p.TASKS.NAMES

        self.backbone = backbone
        self.multi_task_decoder = TransformerDecoder(p) 
        self.heads = heads 

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}

        # Backbone 
        x, selected_fea = self.backbone(x) 
        
        # transformer decoder
        task_features, inter_preds = self.multi_task_decoder(selected_fea)

        # Generate predictions
        out = task_features
        for t in self.tasks:
            out[t] = F.interpolate(self.heads[t](task_features[t]), img_size, mode=INTERPOLATE_MODE)
        out['inter_preds'] = {t: F.interpolate(v, img_size, mode=INTERPOLATE_MODE) for t, v in inter_preds.items()}
            
        return out
