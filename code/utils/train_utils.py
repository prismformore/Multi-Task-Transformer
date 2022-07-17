# Rewritten based on MTI-Net by Hanrong Ye
# Original authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)


import os, json
from evaluation.evaluate_utils import PerformanceMeter
from utils.utils import to_cuda
import torch
from tqdm import tqdm
from utils.test_utils import test_phase
import pdb

# import torch.profiler

def update_tb(tb_writer, tag, loss_dict, iter_no):
    for k, v in loss_dict.items():
        tb_writer.add_scalar(f'{tag}/{k}', v.item(), iter_no)


def train_phase(p, args, train_loader, test_dataloader, model, criterion, optimizer, scheduler, epoch, tb_writer, tb_writer_test, iter_count):
    """ Vanilla training with fixed loss weights """
    model.train() 

    for i, cpu_batch in enumerate(tqdm(train_loader)):
        # Forward pass
        batch = to_cuda(cpu_batch)
        images = batch['image'] 
        output = model(images)
        iter_count += 1
        
        # Measure loss
        loss_dict = criterion(output, batch, tasks=p.TASKS.NAMES)

        if tb_writer is not None:
            update_tb(tb_writer, 'Train_Loss', loss_dict, iter_count)
        
        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()
        scheduler.step()
        
        # end condition
        if iter_count >= p.max_iter:
            print('Max itereaction achieved.')
            # return True, iter_count
            end_signal = True
        else:
            end_signal = False

        # Evaluate
        if end_signal:
            eval_bool = True
        elif iter_count % p.val_interval == 0:
            eval_bool = True 
        else:
            eval_bool = False

        # Perform evaluation
        if eval_bool and args.local_rank == 0:
            print('Evaluate at iter {}'.format(iter_count))
            curr_result = test_phase(p, test_dataloader, model, criterion, iter_count)
            tb_update_perf(p, tb_writer_test, curr_result, iter_count)
            print('Evaluate results at iteration {}: \n'.format(iter_count))
            print(curr_result)
            with open(os.path.join(p['save_dir'], p.version_name + '_' + str(iter_count) + '.txt'), 'w') as f:
                json.dump(curr_result, f, indent=4)

            # Checkpoint after evaluation
            print('Checkpoint starts at iter {}....'.format(iter_count))
            torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'model': model.state_dict(), 
                        'epoch': epoch, 'iter_count': iter_count-1}, p['checkpoint'])
            print('Checkpoint finishs.')
            model.train() # set model back to train status

        if end_signal:
            return True, iter_count

    return False, iter_count


class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
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
    if 'human_parts' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/human_parts_mIoU', curr_result['human_parts']['mIoU'], cur_iter)
    if 'sal' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/sal_maxF', curr_result['sal']['maxF'], cur_iter)
    if 'edge' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/edge_val_loss', curr_result['edge']['loss'], cur_iter)
    if 'normals' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/normals_mean', curr_result['normals']['mean'], cur_iter)