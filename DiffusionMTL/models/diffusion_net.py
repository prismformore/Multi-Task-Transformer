import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from models.diffusion.mtdnet import MTDNet
from models.resnet import BasicBlock

from easydict import EasyDict as edict
INTERPOLATE_MODE = 'bilinear'

class DiffusionNet(nn.Module):
    def __init__(self, p, backbone, backbone_channels, embed_dim=None, heads=None):
        super(DiffusionNet, self).__init__()
        self.tasks = p.TASKS.NAMES

        # Backbone
        self.backbone = backbone
        self.backbone_embed = nn.Conv2d(sum(backbone_channels), embed_dim, 3, padding=1)
        self.preliminary_dec = nn.ModuleDict({task: nn.Sequential(BasicBlock(embed_dim, embed_dim), BasicBlock(embed_dim, embed_dim)) for task in self.tasks})
        self.preliminary_head = nn.ModuleDict({task: nn.Sequential(nn.Conv2d(embed_dim, p.TASKS.NUM_OUTPUT[task], 1)) for task in self.tasks})

        self.diffusion_model = MTDNet(p, embed_dim)

        # Feature aggregation through deeplab heads
        if heads != None:
            self.heads = heads 


    def forward(self, batch):
        x = batch['image'] #.cuda(non_blocking=True)
        img_size = x.size()[-2:]  #  288,384
        info = {'img_size': img_size}

        # Backbone 
        fea_list = self.backbone(x)

        # diffusion net
        target_size = fea_list[0].shape[2:]
        target_size = [int(_size*0.5) for _size in target_size]
        fea_list = [F.interpolate(_fea, size=target_size, mode='bilinear', align_corners=False) for _fea in fea_list]
        x = self.backbone_embed(torch.cat(fea_list, dim=1))

        # preliminary decoding
        inter_feas = {task: self.preliminary_dec[task](x) for task in self.tasks}
        inter_preds = {task: self.preliminary_head[task](inter_feas[task]) for task in self.tasks}
        x = [inter_feas, inter_preds]

        preds, info = self.diffusion_model(x, info, batch)
        out = {'preds': preds}

        if self.training:
            inter_preds = {t: F.interpolate(inter_preds[t], img_size, mode=INTERPOLATE_MODE, align_corners=False) for t in self.tasks}
            out['inter_preds'] = inter_preds
        
        out['info'] = info
        out['layerwise_res_vit'] = None
            
        return out
