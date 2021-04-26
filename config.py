# -*- coding: utf-8 -*-
import argparse
from logging import getLogger

import torch

logger = getLogger('__main__').getChild('config')

arg_group_list = []
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

def add_argument_group(name):
    arg_group = parser.add_argument_group(name)
    arg_group_list.append(arg_group)
    return arg_group

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--pre_trained_network', type=str, default='inceptionv3')

# Controller
net_arg.add_argument('--num_blocks', type=int, default=17, 
                    help='convolutional layer block number')
net_arg.add_argument('--tie_weights', type=str2bool, default=True)
net_arg.add_argument('--controller_hid', type=int, default=100)
net_arg.add_argument('--controller_init_range', type=float, default=0.1)
net_arg.add_argument('--shared_lr_list', type=float, nargs='+',
                    default=[0, 0.01, 0.05, 0.1])


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='cifar100')
data_arg.add_argument('--num_class', type=int, default=100)

# Training / Test parameters
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
learn_arg.add_argument('--data_augmentation', type=str2bool, default=False)
learn_arg.add_argument('--validation_size', type=float, default=0.2)
learn_arg.add_argument('--batch_size', type=int, default=32)
learn_arg.add_argument('--num_workers', type=int, default=0)
learn_arg.add_argument('--test_batch_size', type=int, default=1)
learn_arg.add_argument('--max_epoch', type=int, default=40)
learn_arg.add_argument('--random_policy', type=str2bool, default=False)
# 報酬を正規化するかどうか
#learn_arg.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])

# Controller prameters
net_arg.add_argument('--batch_train', type=str2bool, default=False)
learn_arg.add_argument('--ema_baseline_decay', type=float, default=0.999) # 公式実装0.999, PyTorch実装0.95, Keras実装は0.999
learn_arg.add_argument('--init_baseline', type=str, default='reward')
learn_arg.add_argument('--discount', type=float, default=1.0) # ノイズ除去用
learn_arg.add_argument('--controller_max_step', type=int, default=2000, 
                       help='step for controller parameters')
learn_arg.add_argument('--controller_optim', type=str, default='adam')
learn_arg.add_argument('--controller_lr', type=float, default=3.5e-4,
                       help="will be ignored if --controller_lr_cosine=True")
learn_arg.add_argument('--controller_lr_cosine', type=str2bool, default=False)
learn_arg.add_argument('--controller_lr_max', type=float, default=0.05,
                       help="lr max for cosine schedule")
learn_arg.add_argument('--controller_lr_min', type=float, default=0.001,
                       help="lr min for cosine schedule")

learn_arg.add_argument('--controller_grad_clip', type=float, default=0)
learn_arg.add_argument('--tanh_c', type=float, default=2.5)
learn_arg.add_argument('--softmax_temperature', type=float, default=5.0)
learn_arg.add_argument('--entropy_coeff', type=float, default=0.1) # ENAS論文中のCNN探索でのパラメタ


# Shared parameters
learn_arg.add_argument('--shared_initial_step', type=int, default=0)
learn_arg.add_argument('--shared_train_iteration', type=int, default=10)
learn_arg.add_argument('--shared_valid_iteration', type=int, default=50)
learn_arg.add_argument('--policy_batch_size', type=int, default=10)
#learn_arg.add_argument('--shared_num_sample', type=int, default=1,
#                       help='# of Monte Carlo samples')
learn_arg.add_argument('--shared_optim', type=str, default='sgd')
learn_arg.add_argument('--shared_l2_reg', type=float, default=1e-4)


# Deriving Architectures
learn_arg.add_argument('--derive_num_sample', type=int, default=100)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load_path', type=str, default=None)
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--cuda_num', type=int, default=0)
misc_arg.add_argument('--random_seed', type=int, default=12345)
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=True)

def get_args():
    args, unparsed = parser.parse_known_args()
    if args.num_gpu >0 and torch.cuda.is_available():
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    
    if len(unparsed) > 1:
        print(f'Unparsed args: {unparsed}')

    return args, unparsed

def log_args(args):
    for key, item in args.__dict__.items():
        logger.debug('{} : {}'.format(key, item))
