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
    task_cfg.FLAGVALS = {'image': cv2.INTER_CUBIC}
    task_cfg.INFER_FLAGVALS = {}

    if 'include_semseg' in task_dictionary.keys() and task_dictionary['include_semseg']:
        tmp = 'semseg'
        task_cfg.NAMES.append('semseg')
        if db_name == 'PASCALContext':
            task_cfg.NUM_OUTPUT[tmp] = 21
        elif db_name == 'NYUD':
            task_cfg.NUM_OUTPUT[tmp] = 40
        else:
            raise NotImplementedError
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST 
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_NEAREST

    if 'include_depth' in task_dictionary.keys() and task_dictionary['include_depth']:
        tmp = 'depth'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 1
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR

    if 'include_human_parts' in task_dictionary.keys() and task_dictionary['include_human_parts']:
        # Human Parts Segmentation
        assert(db_name == 'PASCALContext')
        tmp = 'human_parts'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 7
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_NEAREST

    if 'include_sal' in task_dictionary.keys() and task_dictionary['include_sal']:
        # Saliency Estimation
        assert(db_name == 'PASCALContext')
        tmp = 'sal'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 2
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR

    if 'include_normals' in task_dictionary.keys() and task_dictionary['include_normals']:
        # Surface Normals 
        tmp = 'normals'
        assert(db_name in ['PASCALContext', 'NYUD'])
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 3
        task_cfg.FLAGVALS[tmp] = cv2.INTER_CUBIC
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR
    task_cfg.INFER_FLAGVALS['normals'] = cv2.INTER_LINEAR

    if 'include_edge' in task_dictionary.keys() and task_dictionary['include_edge']:
        # Edge Detection
        assert(db_name in ['PASCALContext', 'NYUD'])
        tmp = 'edge'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 1
        task_cfg.FLAGVALS[tmp] = cv2.INTER_NEAREST
        task_cfg.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR
        other_args['edge_w'] = task_dictionary['edge_w']
        other_args['eval_edge'] = False
    task_cfg.INFER_FLAGVALS['edge'] = cv2.INTER_LINEAR

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

    else:
        raise NotImplementedError

    # set log dir
    if cfg['setup'] == 'multi_task':
        output_dir = os.path.join(root_dir, cfg['train_db_name'], cfg['backbone'], cfg['model'])
    else:
        raise NotImplementedError
        

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

    cfg.update(params)

    return cfg
