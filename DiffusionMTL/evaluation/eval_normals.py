import warnings
import cv2
import os.path
import numpy as np
import glob
import math
import torch
import json


def normal_ize(arr):
    arr_norm = np.linalg.norm(arr, ord=2, axis=2)[..., np.newaxis] + 1e-12
    return arr / arr_norm


def eval_normals(loader, folder):

    deg_diff = []
    for i, sample in enumerate(loader):
        if i % 500 == 0:
            print('Evaluating Surface Normals: {} of {} objects'.format(i, len(loader)))

        # Check for valid labels
        label = sample['normals']
        uniq = np.unique(label)
        if len(uniq) == 1 and uniq[0] == 0:
            continue

        # Load result
        filename = os.path.join(folder, sample['meta']['image'] + '.png')
        pred = 2. * cv2.imread(filename).astype(np.float32)[..., ::-1] / 255. - 1
        pred = normal_ize(pred)

        if pred.shape != label.shape:
            warnings.warn('Prediction and ground truth have different size. Resizing Prediction..')
            pred = cv2.resize(pred, label.shape[::-1], interpolation=cv2.INTER_CUBIC)

        valid_mask = (np.linalg.norm(label, ord=2, axis=2) != 0)
        pred[np.invert(valid_mask), :] = 0.
        label[np.invert(valid_mask), :] = 0.
        label = normal_ize(label)

        deg_diff_tmp = np.rad2deg(np.arccos(np.clip(np.sum(pred * label, axis=2), a_min=-1, a_max=1)))
        deg_diff.extend(deg_diff_tmp[valid_mask])

    deg_diff = np.array(deg_diff)
    eval_result = dict()
    eval_result['mean'] = np.mean(deg_diff)
    eval_result['median'] = np.median(deg_diff)
    eval_result['rmse'] = np.mean(deg_diff ** 2) ** 0.5
    eval_result['11.25'] = np.mean(deg_diff < 11.25) * 100
    eval_result['22.5'] = np.mean(deg_diff < 22.5) * 100
    eval_result['30'] = np.mean(deg_diff < 30) * 100

    eval_result = {x: eval_result[x].tolist() for x in eval_result}

    return eval_result

def normalize_tensor(input_tensor, dim):
    norm = torch.norm(input_tensor, p='fro', dim=dim, keepdim=True)
    zero_mask = (norm == 0)
    norm[zero_mask] = 1
    out = input_tensor.div(norm)
    out[zero_mask.expand_as(out)] = 0
    return out

class NormalsMeter(object):
    def __init__(self, ignore_index=255):
        self.sum_deg_diff = 0
        self.total = 0
        self.ignore_index = ignore_index

    @torch.no_grad()
    def update(self, pred, gt):
        pred = pred.permute(0, 3, 1, 2) # [B, C, H, W]
        pred = 2 * pred / 255 - 1 # reverse post-processing
        valid_mask = (gt != self.ignore_index).all(dim=1)

        pred = normalize_tensor(pred, dim=1)
        gt = normalize_tensor(gt, dim=1)
        deg_diff = torch.rad2deg(2 * torch.atan2(torch.norm(pred - gt, dim=1), torch.norm(pred + gt, dim=1)))
        deg_diff = torch.masked_select(deg_diff, valid_mask)

        self.sum_deg_diff += torch.sum(deg_diff).cpu().item()
        self.total += deg_diff.numel()

    def get_score(self, verbose=False):
        eval_result = dict()
        eval_result['mean'] = self.sum_deg_diff / self.total

        return eval_result


def eval_normals_predictions(database, save_dir, overfit=False):
    """ Evaluate the normals maps that are stored in the save dir """

    # Dataloaders
    if database == 'PASCALContext':
        from data.pascal_context import PASCALContext
        gt_set = 'val'
        db = PASCALContext(split=gt_set, do_edge=False, do_human_parts=False, do_semseg=False,
                                          do_normals=True, overfit=overfit)
    elif database == 'NYUD':
        from data.nyud import NYUD_MT
        gt_set = 'val'
        db = NYUD_MT(split=gt_set, do_normals=True, overfit=overfit)

    else:
        raise NotImplementedError

    base_name = database + '_' + 'test' + '_normals'
    fname = os.path.join(save_dir, base_name + '.json')

    # Eval the model
    print('Evaluate the saved images (surface normals)') 
    eval_results = eval_normals(db, os.path.join(save_dir, 'normals'))
    with open(fname, 'w') as f:
        json.dump(eval_results, f)

    # Print results
    print('Results for Surface Normal Estimation')
    for x in eval_results:
        spaces = ""
        for j in range(0, 15 - len(x)):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_results[x]))

    return eval_results
