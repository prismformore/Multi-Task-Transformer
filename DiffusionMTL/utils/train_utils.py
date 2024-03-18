from evaluation.evaluate_utils import PerformanceMeter
from utils.utils import get_output, to_cuda, mkdir_if_missing
import torch
from tqdm import tqdm
import pdb

import os, json, imageio
import numpy as np

from utils.test_utils import test_phase


def update_tb(tb_writer, tag, loss_dict, iter_no):
    for k, v in loss_dict.items():
        if type(v) == torch.Tensor:
            tb_writer.add_scalar(f'{tag}/{k}', v.item(), iter_no)
        elif type(v) == int:
            tb_writer.add_scalar(f'{tag}/{k}', v, iter_no)


def train_phase(p, args, train_loader, test_dataloader, model, criterion, optimizer, scheduler, epoch, tb_writer, tb_writer_test, iter_count):
    """ Vanilla training with fixed loss weights """

    if iter_count >= p.max_iter:
        print('Max itereaction achieved.')
        return True, iter_count

    model.train()

    for i, cpu_batch in enumerate(tqdm(train_loader)):
        # Forward pass
        batch = to_cuda(cpu_batch)
        output = model(batch)
        iter_count += 1
        
        # Measure loss
        loss_dict = criterion(output, batch, tasks=p.TASKS.NAMES)

        if tb_writer is not None:
            update_tb(tb_writer, 'Train_Loss', loss_dict, iter_count)
        
        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), **p.grad_clip_param)
        optimizer.step()

        # end condition
        if iter_count >= p.max_iter:
            print('Max itereaction achieved.')
            end_signal = True
        else:
            end_signal = False

        # Evaluate
        # Don't eval at every epoch
        if end_signal:
            eval_bool = True
        elif iter_count % p.val_interval == 0:
            eval_bool = True 
        else:
            eval_bool = False

        # Perform evaluation
        if eval_bool and args.local_rank == 0:
            print('Evaluate at iter {}'.format(iter_count))
            if end_signal:
                curr_result = test_phase(p, test_dataloader, model, criterion, iter_count, save_result=True)
            else:
                curr_result = test_phase(p, test_dataloader, model, criterion, iter_count)
            tb_update_perf(p, tb_writer_test, curr_result, iter_count)
            print('Evaluate results at iteration {}: \n'.format(iter_count))
            print(curr_result)
            with open(os.path.join(p['save_dir'], p.version_name + '_' + str(iter_count) + '.txt'), 'w') as f:
                json.dump(curr_result, f, indent=4)

            model.train()
            # Checkpoint after evaluation
            print('Checkpoint ...')
            torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'model': model.state_dict(), 
                        'epoch': epoch, 'iter_count': iter_count-1}, p['checkpoint'])
        if end_signal:
            return True, iter_count
    return False, iter_count


class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    '''Borrow from ATRC code'''
    def __init__(self, optimizer, max_iterations, gamma=0.9, min_lr=0., last_epoch=-1):
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # slight abuse: last_epoch refers to last iteration
        factor = (1 - self.last_epoch /
                  float(self.max_iterations)) ** self.gamma
        return [(base_lr - self.min_lr) * factor + self.min_lr for base_lr in self.base_lrs]

def tb_update_perf(p, tb_writer_test, curr_result, cur_iter):
    if 'semseg' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/semseg_miou', curr_result['semseg']['mIoU'], cur_iter)
    if 'depth' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/depth_rmse', curr_result['depth']['rmse'], cur_iter)
        tb_writer_test.add_scalar('perf/depth_logrmse', curr_result['depth']['log_rmse'], cur_iter)
        tb_writer_test.add_scalar('perf/abs_rel', curr_result['depth']['abs_rel'], cur_iter)
        tb_writer_test.add_scalar('perf/sq_rel', curr_result['depth']['sq_rel'], cur_iter)
        tb_writer_test.add_scalar('perf/abs_err', curr_result['depth']['abs_err'], cur_iter)
    if 'human_parts' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/human_parts_mIoU', curr_result['human_parts']['mIoU'], cur_iter)
    if 'sal' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/sal_maxF', curr_result['sal']['maxF'], cur_iter)
    if 'edge' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/edge_val_loss', curr_result['edge']['loss'], cur_iter)
    if 'normals' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/normals_mean', curr_result['normals']['mean'], cur_iter)
