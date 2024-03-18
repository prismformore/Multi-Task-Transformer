
import argparse
import cv2
import os
import numpy as np
import sys, json
import torch

from utils.utils import mkdir_if_missing
from utils.config import create_config
from utils.common_config import get_train_dataset, get_transformations,\
                                get_test_dataset, get_train_dataloader, get_test_dataloader,\
                                get_optimizer, get_model, get_criterion
from evaluation.evaluate_utils import PerformanceMeter
from utils.logger import Logger
from utils.train_utils import train_phase
from utils.test_utils import test_phase
from termcolor import colored

from torch.utils.tensorboard import SummaryWriter
import time
start_time = time.time()

# DDP
import torch.distributed as dist
import datetime
dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(0, 3600*2))

# Parser
parser = argparse.ArgumentParser(description='Vanilla Training')
parser.add_argument('--config_exp',
                    help='Config file for the experiment', default='./configs/cityscapes/hrnet18/multi_task_baseline.yml')
parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()
# args.local_rank = int(os.environ['LOCAL_RANK'])

print('local rank: %s' %args.local_rank)
torch.cuda.set_device(args.local_rank)

# CUDNN
torch.backends.cudnn.benchmark = True
import pdb

def set_seed(seed):
    # Stop randomness in each process in the DDP
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(params):
    set_seed(0)
    # Retrieve config file
    cv2.setNumThreads(0)
    p = create_config(args.config_exp, params)

    sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'))
    if args.local_rank == 0:
        print(colored(p, 'red'))

    # tensorboard
    tb_log_dir = '../' + p.version_name + '/tb_dir' #os.path.join(p['output_dir'], 'tensorboard_logdir')
    p.tb_log_dir = tb_log_dir
    if args.local_rank == 0:
        train_tb_log_dir = tb_log_dir + '/train'
        test_tb_log_dir = tb_log_dir + '/test'
        if params['run_mode'] != 'infer':
            mkdir_if_missing(tb_log_dir)
            mkdir_if_missing(train_tb_log_dir)
            mkdir_if_missing(test_tb_log_dir)
        tb_writer_train = SummaryWriter(train_tb_log_dir)
        tb_writer_test = SummaryWriter(test_tb_log_dir)
        print(f"Tensorboard dir: {tb_log_dir}")
    else:
        tb_writer_train = None
        tb_writer_test = None

    # Init performance meter
    _ = PerformanceMeter(p, [t for t in p.TASKS.NAMES if t != '3ddet'])

    # Get model
    model = get_model(p)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    # model = torch.nn.DataParallel(model)
    # model = model.cuda()

    # Get criterion
    criterion = get_criterion(p)

    # Optimizer
    scheduler, optimizer = get_optimizer(p, model)

    # Transforms 
    if p.train_db_name != 'NYUD':
        train_transforms, val_transforms = get_transformations(p)
    else:
        train_transforms, val_transforms = None, None
    if params['run_mode'] != 'infer':
        train_dataset = get_train_dataset(p, train_transforms)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        train_dataloader = get_train_dataloader(p, train_dataset, train_sampler)
    test_dataset = get_test_dataset(p, val_transforms)
    test_dataloader = get_test_dataloader(p, test_dataset)

    # Resume from checkpoint
    if os.path.exists(p['checkpoint']) or params['run_mode'] == 'infer':
        if args.local_rank == 0:
            print(colored('Restart from checkpoint {}'.format(p['checkpoint']), 'blue'))
        checkpoint = torch.load(p['checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        if True:
            # filter model state dict
            saved_model_state_dict = checkpoint['model']
            now_model_state_dict = model.state_dict()
            def del_different_shape(saved, now):
                new_state_dict = {}
                for key, val in saved.items():
                    if val.shape != now[key].shape:
                        print(f'Warning: Model Loading: Skipping {key}')
                        pass
                    else:
                        new_state_dict[key] = val
                return new_state_dict
            model_state_dict = del_different_shape(saved_model_state_dict, now_model_state_dict)
        else:
            model_state_dict = checkpoint['model']
        model.load_state_dict(model_state_dict, strict=False)
        start_epoch = checkpoint['epoch'] + 1
        iter_count  = checkpoint['iter_count'] # already + 1 when saving
    else:
        if args.local_rank == 0:
            print(colored('Fresh start...', 'blue'))
        start_epoch = 0
        iter_count = 0

    if DEBUG_FLAG and args.local_rank == 0:
        print("\nFirst Testing...")
        eval_test = test_phase(p, test_dataloader, model, criterion, iter_count)
        print(eval_test)
    
    # Main loop
    if params['run_mode'] != 'infer':
        for epoch in range(start_epoch, p['epochs']):
            train_sampler.set_epoch(epoch)
            if args.local_rank == 0:
                print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
                print(colored('-'*10, 'yellow'))

            end_signal, iter_count = train_phase(p, args, train_dataloader, test_dataloader, model, criterion, optimizer, scheduler, epoch, tb_writer_train, tb_writer_test, iter_count)
            scheduler.step()

            if end_signal:
                break

    # Evaluate best model at the end
    # running eval
    if args.local_rank == 0:
        if p.run_mode == 'infer' or True:
            # print('Infer at batch {}'.format(start_epoch))
            if p.run_mode == 'train':
                print('Checkpoint ...')
                torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'model': model.state_dict(), 
                            'epoch': epoch, 'iter_count': iter_count-1}, p['checkpoint'])
            print('Infer at iteration {}'.format(iter_count))
            eval_test = test_phase(p, test_dataloader, model, criterion, iter_count, save_result=True)
            print('Infer test restuls:')
            print(eval_test)

        end_time = time.time()
        run_time = (end_time-start_time) / 3600
        print('Total running time: {} h.'.format(run_time))

if __name__ == "__main__":
    params = {}
    params['version_name'] = 'SSdiffMTLfixL_pascal_res18_v1_NoiseFea_CACondReplaceTranspose_woFeatCat_weakerNoise_os'

    # IMPORTANT VARIABLES
    params["semseg_save_train_class"] = False
    params['run_mode'] = 'train'

    DEBUG_FLAG = False 

    args.config_exp = './configs/pascal/config.yml'
    main(params)
