# By Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from evaluation.evaluate_utils import PerformanceMeter
from tqdm import tqdm
from utils.utils import get_output, mkdir_if_missing
from evaluation.evaluate_utils import save_model_pred_for_one_task
import torch
import os

@torch.no_grad()
def test_phase(p, test_loader, model, criterion, epoch):
    all_tasks = [t for t in p.TASKS.NAMES]
    two_d_tasks = [t for t in p.TASKS.NAMES if t != '3ddet']
    performance_meter = PerformanceMeter(p, two_d_tasks)

    model.eval()

    tasks_to_save = []
    if 'edge' in all_tasks:
        tasks_to_save.append('edge')
    if '3ddet' in all_tasks:
        tasks_to_save.append('3ddet')

    save_dirs = {task: os.path.join(p['save_dir'], task) for task in tasks_to_save}
    for save_dir in save_dirs.values():
        mkdir_if_missing(save_dir)
    
    for i, batch in enumerate(tqdm(test_loader)):
        # Forward pass
        with torch.no_grad():
            images = batch['image'].cuda(non_blocking=True)
            targets = {task: batch[task].cuda(non_blocking=True) for task in two_d_tasks}

            output = model.module(images) # to make ddp happy
        
            # Measure loss and performance
            performance_meter.update({t: get_output(output[t], t) for t in two_d_tasks}, 
                                    {t: targets[t] for t in two_d_tasks})

            for task in tasks_to_save:
                save_model_pred_for_one_task(p, i, batch, output, save_dirs, task, epoch=epoch)


    eval_results = performance_meter.get_score(verbose = True)

    if '3ddet' in all_tasks:
        from detection_toolbox.det_eval import eval_3ddet
        saved_eval_test = eval_3ddet(p, save_dir=p['save_dir'])
        interest_metric = p.get_CS_metrics_of_interest(saved_eval_test)
        eval_results['3ddet_brief'] = interest_metric
        eval_results['3ddet'] = saved_eval_test

    return eval_results


@torch.no_grad()
def vis_phase(p, test_loader, model, criterion, epoch):
    from utils.visualization_utils import vis_pred_for_one_task
    # Originally designed for visualization on cityscapes-3D
    model.eval()

    tasks_to_save = ['3ddet', 'semseg', 'depth']

    save_dirs = {task: os.path.join(p['save_dir'], 'vis', task) for task in tasks_to_save}
    for save_dir in save_dirs.values():
        mkdir_if_missing(save_dir)
    
    for i, batch in enumerate(tqdm(test_loader)):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        output = model.module(images) # to make ddp happy
    
        for task in tasks_to_save:
            vis_pred_for_one_task(p, batch, output, save_dirs[task], task)
        del batch, output, images


    return
