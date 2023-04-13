# By Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import matplotlib.pyplot as plt
from PIL import Image
import imageio, os, cv2
import numpy as np
from utils.utils import get_output
import torch.nn.functional as F
import torch
import warnings
from concurrent.futures import ThreadPoolExecutor

def create_cityscapes_label_colormap():
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    return colormap


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype = np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ ( np.uint8(str_id[-1]) << (7-j))
            g = g ^ ( np.uint8(str_id[-2]) << (7-j))
            b = b ^ ( np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap

def vis_semseg(p, _semseg):
    if p['train_db_name'] == "NYUD":
        new_cmap = labelcolormap(40)
    elif p['train_db_name'] == "PASCALContext":
        new_cmap = labelcolormap(21)
    elif p['train_db_name'] == "Cityscapes3D":
        new_cmap = create_cityscapes_label_colormap()
    _semseg = new_cmap[_semseg]  
    return _semseg

def vis_parts(inp):
    new_cmap = labelcolormap(7)
    inp = new_cmap[inp]  
    return inp


@torch.no_grad()
def vis_pred_for_one_task(p, sample, output, save_dir, task):
    inputs, meta = sample['image'], sample['meta']
    bs = int(inputs.size()[0])

    if task == '3ddet':
        from detection_toolbox.det_tools import bbox2fig
        det_res_list = get_output(output[task], task, p=p, label=sample)
        cam_params = [{k: v[sa] for k, v in sample['bbox_camera_params'].items()} for sa in range(bs)]

        if False:
            # Serial
            for jj in range(bs):
                K_matrixes = sample['meta']['K_matrix'] # numpy
                vis_fname = meta['img_name'][jj]
                box_no = len(det_res_list[jj]['img_bbox']['scores_3d'])
                gt_labels = None
                vis_pred = bbox2fig(p, inputs[jj].cpu(), det_res_list[jj], K_matrixes[jj], cam_params[jj], gt_labels)
                vis_pred = vis_pred.astype(np.uint8)
                filename = '{}_{}.png'.format(vis_fname, box_no)
                filepath = os.path.join(save_dir, filename)
                plt.imsave(filepath, vis_pred)
        else:
            # Parallel
            def save_visualization(args):
                jj, bs, sample, meta, inputs, det_res_list, cam_params, save_dir = args
                K_matrixes = sample['meta']['K_matrix']  # numpy
                vis_fname = meta['img_name'][jj]
                box_no = len(det_res_list[jj]['img_bbox']['scores_3d'])
                gt_labels = None
                vis_pred = bbox2fig(p, inputs[jj].cpu(), det_res_list[jj], K_matrixes[jj], cam_params[jj], gt_labels)
                vis_pred = vis_pred.astype(np.uint8)
                filename = '{}_{}.png'.format(vis_fname, box_no)
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, vis_pred[:, :, [2, 1, 0]])  # Convert RGB to BGR for OpenCV
            with ThreadPoolExecutor() as executor:
                args = [(jj, bs, sample, meta, inputs, det_res_list, cam_params, save_dir) for jj in range(bs)]
                futures = [executor.submit(save_visualization, arg) for arg in args]
                _ = [future.result() for future in futures]

        return

    warnings.warn('Warning: We assume all the images have the same size!!!')
    im_height = meta['img_size'][0][0]
    im_width = meta['img_size'][0][1]    
    if task == 'semseg':
        output_task = F.interpolate(output[task], (im_height, im_width), mode='bilinear')
        # During visualization, here we always use the train class to draw prediction (totally 19)
        output_task = get_output(output_task, task).cpu().data.numpy()
    else:
        output_task = F.interpolate(output[task], (im_height, im_width), mode='bilinear')
        output_task = get_output(output_task, task).cpu().data.numpy()

    if False: 
        # Serial
        for jj in range(int(inputs.size()[0])):
            im_name = meta['img_name'][jj]
            pred = output_task[jj] # (H, W) or (H, W, C)

            # visualize result 
            arr = pred # (H, W, (C))
            if task == 'semseg':
                arr = vis_semseg(p, arr)
            elif task == 'sal':
                pass
            elif task == 'edge':
                pass
            elif task == 'human_parts':
                arr = vis_parts(arr)
            elif task == 'normals':
                pass
            elif task == 'depth':
                arr = arr.squeeze()
                arr = arr ** 0.15 # NEED TO take a "root" of the values because the original one varaince is so large that the visualization is bad.
                plt.imsave(os.path.join(save_dir, '{}_{}.png'.format(im_name, task)), arr, cmap='jet')
                continue
            arr_uint8 = arr.astype(np.uint8)
            filename = '{}_{}.png'.format(im_name, task)
            filepath = os.path.join(save_dir, filename)
            plt.imsave(filepath, arr_uint8)
    else:
        # parallel
        def save_image(meta, output_task, save_dir, task, idx):
            im_name = meta['img_name'][idx]
            pred = output_task[idx]

            arr = pred
            if task == 'semseg':
                arr = vis_semseg(p, arr)
            elif task == 'sal':
                pass
            elif task == 'edge':
                pass
            elif task == 'human_parts':
                arr = vis_parts(arr)
            elif task == 'normals':
                pass
            elif task == 'depth':
                arr = arr.squeeze()
                arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
                arr_colored = cv2.applyColorMap((arr).astype(np.uint8), cv2.COLORMAP_JET)
                filepath = os.path.join(save_dir, '{}_{}.png'.format(im_name, task))
                cv2.imwrite(filepath, arr_colored)
                return

            arr_uint8 = arr.astype(np.uint8)
            if arr_uint8.ndim == 3:
                arr_uint8 = arr_uint8[:, :, [2, 1, 0]] # Convert RGB to BGR for OpenCV
            # else:
            #     arr_uint8 = cv2.applyColorMap(arr_uint8, cv2.COLORMAP_JET)
            filename = '{}_{}.png'.format(im_name, task)
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, arr_uint8) 

        def save_images_in_parallel(meta, output_task, save_dir, task):
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(save_image, meta, output_task, save_dir, task, idx) for idx in range(int(inputs.size()[0]))]
                _ = [future.result() for future in futures]

        save_images_in_parallel(meta, output_task, save_dir, task)
    return
