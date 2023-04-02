# By Hanrong Ye

# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import os
import cv2
import yaml
from easydict import EasyDict as edict
from utils.utils import mkdir_if_missing
import pdb


def parse_task_dictionary(db_name, task_dictionary):
    """ 
        Return a dictionary with task information. 
        Additionally we return a dict with key, values to be added to the main dictionary
    """

    task_cfg = edict()
    other_args = dict()
    task_cfg.NAMES = []
    task_cfg.NUM_OUTPUT = {}

    if 'include_semseg' in task_dictionary.keys() and task_dictionary['include_semseg']:
        tmp = 'semseg'
        task_cfg.NAMES.append('semseg')
        if db_name == 'PASCALContext':
            task_cfg.NUM_OUTPUT[tmp] = 21
        elif db_name == 'NYUD':
            task_cfg.NUM_OUTPUT[tmp] = 40
        elif db_name == 'Cityscapes3D':
            task_cfg.NUM_OUTPUT[tmp] = 19
        else:
            raise NotImplementedError

    if 'include_depth' in task_dictionary.keys() and task_dictionary['include_depth']:
        tmp = 'depth'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 1
        # Set effective depth evaluation range. Refer to:
        # https://github.com/sjsu-smart-lab/Self-supervised-Monocular-Trained-Depth-Estimation-using-Self-attention-and-Discrete-Disparity-Volum/blob/3c6f46ab03cfd424b677dfeb0c4a45d6269415a9/evaluate_city_depth.py#L55
        task_cfg.depth_max = 80.0
        task_cfg.depth_min = 0.

    if 'include_human_parts' in task_dictionary.keys() and task_dictionary['include_human_parts']:
        # Human Parts Segmentation
        assert(db_name == 'PASCALContext')
        tmp = 'human_parts'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 7

    if 'include_sal' in task_dictionary.keys() and task_dictionary['include_sal']:
        # Saliency Estimation
        assert(db_name == 'PASCALContext')
        tmp = 'sal'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 2

    if 'include_normals' in task_dictionary.keys() and task_dictionary['include_normals']:
        # Surface Normals 
        tmp = 'normals'
        assert(db_name in ['PASCALContext', 'NYUD'])
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 3

    if 'include_edge' in task_dictionary.keys() and task_dictionary['include_edge']:
        # Edge Detection
        assert(db_name in ['PASCALContext', 'NYUD'])
        tmp = 'edge'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 1
        other_args['edge_w'] = task_dictionary['edge_w']

    if 'include_3ddet' in task_dictionary.keys() and task_dictionary['include_3ddet']:
        # Depth
        tmp = '3ddet'
        task_cfg.NAMES.append(tmp)
        if db_name == 'Cityscapes3D':
            task_cfg.NUM_OUTPUT[tmp] = 12+6
        else:
            raise NotImplementedError

    return task_cfg, other_args


def create_config(exp_file, params):
    
    with open(exp_file, 'r') as stream:
        config = yaml.safe_load(stream)

    # Copy all the arguments
    cfg = edict()
    for k, v in config.items():
        cfg[k] = v

    # set root dir
    root_dir = cfg["out_dir"] + cfg['version_name']

    # Parse the task dictionary separately
    cfg.TASKS, extra_args = parse_task_dictionary(cfg['train_db_name'], cfg['task_dictionary'])

    for k, v in extra_args.items():
        cfg[k] = v
    
    # Other arguments 
    if cfg['train_db_name'] == 'PASCALContext':
        cfg.TRAIN = edict()
        cfg.TRAIN.SCALE = (512, 512)
        cfg.TEST = edict()
        cfg.TEST.SCALE = (512, 512)

    elif cfg['train_db_name'] == 'NYUD':
        cfg.TRAIN = edict()
        cfg.TEST = edict()
        cfg.TRAIN.SCALE = (448, 576)
        cfg.TEST.SCALE = (448, 576)

    elif cfg['train_db_name'] == 'Cityscapes3D':
        cfg.IMAGE_ORI_SIZE = (1024, 2048)
        cfg.TRAIN = edict()
        cfg.TRAIN.SCALE = (1024, 2048)
        cfg.TEST = edict()
        cfg.TEST.SCALE = (1024, 2048) # original size

    else:
        raise NotImplementedError

    # set log dir
    output_dir = root_dir
    cfg['root_dir'] = root_dir
    cfg['output_dir'] = output_dir
    cfg['save_dir'] = os.path.join(output_dir, 'results')
    cfg['checkpoint'] = os.path.join(output_dir, 'checkpoint.pth.tar')
    if params['run_mode'] != 'infer':
        mkdir_if_missing(cfg['output_dir'])
        mkdir_if_missing(cfg['save_dir'])

    from configs.mypath import db_paths, PROJECT_ROOT_DIR
    params['db_paths'] = db_paths
    params['PROJECT_ROOT_DIR'] = PROJECT_ROOT_DIR

    # add fcos3D detection head params
    if '3ddet' in cfg.TASKS.NAMES:
        from configs.cityscapes3d.det_head_params import det_model_params, det_head_params, \
             get_CS_metrics_of_interest
        from detection_toolbox.det_model import DetModel
        # check if 3d related packages installed properly at the beginning
        from detection_toolbox.det_tools import bbox2json, bbox2fig

        # adjust stride
        strides = det_model_params.strides
        ds_ratio = cfg.IMAGE_ORI_SIZE[0] // cfg.TRAIN.SCALE[0]
        strides = [_str*ds_ratio for _str in strides] 
        det_model_params.strides = [_ / cfg.img_ds_ratio for _ in strides] # Adjust the strides based on image resize in the model

        cfg.det_model_params = det_model_params
        cfg.detmodel = DetModel(**det_model_params)
        cfg.det_head_params = det_head_params
        cfg.get_CS_metrics_of_interest = get_CS_metrics_of_interest

    cfg.update(params)

    return cfg
