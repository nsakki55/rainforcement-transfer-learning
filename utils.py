# -*- coding: utf-8 -*-
import os
import json
import csv

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from datetime import datetime
from collections import defaultdict
from logging import getLogger, Formatter, StreamHandler, FileHandler, INFO, DEBUG

logger = getLogger('__main__').getChild('utils')

def get_logger(log_path, name=__file__, level=INFO):
    name = os.path.basename(name)
    logger = getLogger(name)
    
    # 初期設定が終わってる場合
    if getattr(logger, '_init_done_', None):
        logger.setLevel(DEBUG)
        return logger
    
    logger._init_done = True
    logger.propagate = False
    logger.setLevel(DEBUG)

    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    str_handler = StreamHandler()
    str_handler.setFormatter(formatter)
    str_handler.setLevel(level)

    file_handler = FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(DEBUG)

    logger.addHandler(str_handler)
    logger.addHandler(file_handler)

    logger.info('logger is initialized, log path: {}'.format(log_path))
    return logger


def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def prepare_dirs(args):

    logger.info('prepare direcotry')
    #args.model_name = "{}_{}_{}".format(args.dataset, args.pre_trained_network, get_time())
    args.model_name = "{}_{}_{}".format(args.dataset, args.pre_trained_network, get_time())
    if not hasattr(args, 'model_dir'):
        args.model_dir = os.path.join(args.log_dir, args.model_name)
        args.log_path = os.path.join(args.model_dir, args.model_name + '.log')
    
    for path in [args.log_dir, args.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


class KeyDefaultDict(defaultdict):
    # https://docs.python.org/ja/3/reference/datamodel.html#object.__missing__
    def __missing__(self, key):
        # https://docs.python.org/ja/3/library/collections.html
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

# 入力変数をTensor型にして、自動微分できるようにVariable型に変換する
def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out

def save_args(args):
    param_path = os.path.join(args.model_dir, 'params.json')
    logger.info("[*] MODEL dir: {}".format(args.model_dir))
    logger.info("[*] PARAM path: {}".format(param_path))

    with open(param_path, 'w') as fp:
       json.dump(args.__dict__, fp, indent=4, sort_keys=True)

def log_optimizer_lr(optimizer):
    lr_list = []
    for group in optimizer.param_groups:
        lr_list.append(group['lr'])
    logger.debug('learning rate:{}'.format(lr_list))

def get_random_policy(policy_length, lr_count):
        policy_nums = [i for i in range(lr_count)]
        return np.random.choice(policy_nums, policy_length)

class ResultContainer(object):
    def __init__(self, columns, dir_path, file_name):
        self.columns = columns
        self.file_path = os.path.join(dir_path, file_name)
        with open(self.file_path, 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(self.columns)
            
        logger.debug('result container initialize: {}'.format(columns))
        
    def add(self, values):
        try:
            with open(self.file_path, 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(values)    
        except:
            raise ValueError('Not matched with column: {}'.format(self.columns))
