from evaluation.evaluate_utils import PerformanceMeter
from tqdm import tqdm
from utils.utils import get_output, to_cuda
from evaluation.evaluate_utils import mkdir_if_missing, save_model_pred_for_one_task
import torch
import os

@torch.no_grad()
def test_phase(p, test_loader, model, criterion, iter_count, save_result=False):
    all_tasks = [t for t in p.TASKS.NAMES]
    tasks = [t for t in p.TASKS.NAMES if t != '3ddet']
    if len(all_tasks) == 0:
        return {}

    performance_meter = PerformanceMeter(p, tasks)

    model.eval()

    tasks_to_save = []
    if 'edge' in p.TASKS.NAMES:
        tasks_to_save.append('edge')
    save_dirs = {task: os.path.join(p['save_dir'], task) for task in tasks_to_save}
    for save_dir in save_dirs.values():
        mkdir_if_missing(save_dir)
    
    for i, cpu_batch in enumerate(tqdm(test_loader)):
        # Forward pass
        batch = to_cuda(cpu_batch)

        output = model(batch)
        
        # Measure loss and performance
        performance_meter.update({t: get_output(output['preds'][t], t) for t in tasks}, 
                                 {t: batch[t] for t in tasks})

        if save_result:
            for task in tasks_to_save:
                save_model_pred_for_one_task(p, batch, output['preds'], save_dirs, task, epoch=iter_count)


    eval_results = performance_meter.get_score(verbose = True)

    return eval_results
