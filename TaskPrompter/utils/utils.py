# Rewritten based on MTI-Net and ATRC by Hanrong Ye
# Original authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import torch
import torch.nn.functional as F
import pdb

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            pass

# get_cityscapes_class
CS_valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
CS_class_map = dict(zip(range(19), CS_valid_classes))
def get_cityscapes_class(mask_inp):
    mask = mask_inp.detach().clone()
    for _validc in range(19):
        mask[mask_inp==_validc] = CS_class_map[_validc]
    return mask


def get_output(output, task, p=None, label=None, semseg_save_train_class=True):
    
    if task == 'normals':
        output = output.permute(0, 2, 3, 1)
        output = (F.normalize(output, p = 2, dim = 3) + 1.0) * 255 / 2.0
    
    elif task in {'semseg'}:
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)

        # get original class index in Cityscapes3D dataset
        if not semseg_save_train_class:
            output = get_cityscapes_class(output)

    elif task in {'human_parts'}:
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)
    
    elif task in {'edge'}:
        output = output.permute(0, 2, 3, 1)
        output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)), dim=3)

    elif task in {'sal'}:
        output = output.permute(0, 2, 3, 1)
        output = F.softmax(output, dim=3)[:, :, :, 1] *255 # torch.squeeze(255 * 1 / (1 + torch.exp(-output)))
    
    elif task in {'depth'}:
        output.clamp_(min=0.)
        output = output.permute(0, 2, 3, 1)

    elif task in {'3ddet'}:
        output = p.detmodel.get_results_from_bbox(output, label, rescale=False)
    
    else:
        raise ValueError('Select one of the valid tasks')

    return output

def to_cuda(batch):
    if type(batch) == dict:
        out = {}
        for k, v in batch.items():
            if k == 'meta':
                out[k] = v
            else:
                out[k] = to_cuda(v)
        return out
    elif type(batch) == torch.Tensor:
        return batch.cuda(non_blocking=True)
    elif type(batch) == list:
        return [to_cuda(v) for v in batch]
    else:
        return batch

# From PyTorch internals
import collections.abc as container_abcs
from itertools import repeat
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_2tuple = _ntuple(2)