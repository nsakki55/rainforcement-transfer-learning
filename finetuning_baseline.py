#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import random
import argparse
import time
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from logging import getLogger, DEBUG, INFO, StreamHandler, FileHandler, Formatter

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms, datasets


# 乱数シード
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

def main(args):

    time_start = time.time()

    net = initialize_network(num_classes=100, pretrained=True)

    size = 299
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
        
    transforms_dict = {'train' : transforms.Compose([
                    transforms.Resize(size),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ]),
                    'test' : transforms.Compose([
                    transforms.Resize(size),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ])}

    train_datasets = datasets.ImageFolder(root='../data/cifar100/train', 
                                          transform=transforms_dict['train']) 

    test_datasets = datasets.ImageFolder(root='../data/cifar100/test',
                                        transform=transforms_dict['test'])

    dataloaders_dict = {'train' : data.DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True),
                        'test' : data.DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False)}        
    # 損失関数
    criterion = nn.CrossEntropyLoss()

    df_result, net_trained = train_model(net, dataloaders_dict, criterion)

    df_result.to_csv(f'inception_v3_cifar100_{args.result_name}.csv', index=False)
    logger.info(f'inception_v3_cifar100_{args.result_name}.csv result is saved')

    torch.save(net_trained.state_dict(), f'inception_v3_cifar100_{args.result_name}_weights.pth')
    logger.info(f'inception_v3_cifar100_{args.result_name}_weights.pth network parameter is saved')

    time_elapsed = time.time() - time_start
    logger.info('Experiment finish in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def initialize_network(num_classes, pretrained=True):
    '''
    学習済みモデルを読み込む
    '''

    net = models.inception_v3(pretrained=pretrained,aux_logits=False)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    net.train()

    logger.info('initialize network, pretrained = {}, output_features : {}'.format(pretrained, num_classes))
    return net

def _summarize_train(total_loss, total_acc, log_step, epoch):
    cur_loss = (total_loss / log_step).item()

    print(f'| epoch {epoch:3d}'
            f'| loss {cur_loss:.2f}'
            f'| acc {total_acc:.3f}')

def train_model(net, dataloaders_dict, criterion, max_epochs=20):
    # PyTorchの転移学習チュートリアルにあるInceptionV3の扱いを参考
    #https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

    time_start = time.time()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.backends.cudnn.benchmark = True

    results = {'epochs' : list(),
               'train_loss' : list(),
               'test_loss' : list(),
               'train_acc' : list(),
               'test_acc' : list()}

    logger.info('Training START, device:{}'.format(device)) 

    for epoch in range(max_epochs):
        
        logger.info('Epoch {}/{}'.format(epoch+1, max_epochs))
        results['epochs'].append(epoch + 1)
        
        optimizer = optim.Adam(params=net.parameters())
                
        net.to(device)

        for phase in ['train', 'test']:
            if phase == 'train':
                net.train() 
            else:
                net.eval()
            
            epoch_loss = 0.0
            epoch_corrects = 0.0
            
            max_step = 1000
            log_step = 50
            step = 0
            total_loss = 0
            total_count = 0
            correct_count = 0

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                if step > max_step:
                    break    
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs.data, 1) # 出力の最大値のインデックス番号を予測とする

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    total_loss += loss
                    correct = (preds == labels).sum().item()
                    
                    correct_count += correct
                    total_count += labels.size(0)
                    total_acc = correct_count / total_count
                    print('{}/{}'.format(correct_count, total_count))
                    print(total_acc)
                    if ((step % log_step) == 0) and (step > 0):
                        _summarize_train(total_loss, total_acc, epoch, log_step)
                        print('{}/{}'.format(correct_count, total_count))

                        total_loss = 0
                        total_acc = 0
                        correct_count = 0
                        total_count = 0
                    
                    step += 1
                epoch_loss += loss.item() * inputs.size(0) # ミニバッチの合計損失
                epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            
            results[f'{phase}_loss'].append(epoch_loss)
            results[f'{phase}_acc'].append(epoch_acc.item())
 
            logger.info('{} Loss : {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    df_result = pd.DataFrame(results)

    time_elapsed = time.time() - time_start
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return df_result, net
      


def setup_logger(name, logfile='LOGFILENAME.txt'):
    '''
    Logger の作成
    参考；https://docs.python.org/ja/3/howto/logging-cookbook.html#logging-cookbook
    '''
    logger = getLogger(name)
    logger.setLevel(DEBUG)

    # create file handler which logs even DEBUG messages
    log_path = os.path.join('logs', logfile)
    fh = FileHandler(log_path)
    fh.setLevel(DEBUG)
    fh_formatter = Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(name)s -> %(message)s')
    fh.setFormatter(fh_formatter)

    # create console handler with a INFO log level
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch_formatter = Formatter('%(asctime)s -> %(message)s', '%Y-%m-%d %H:%M:%S')
    ch.setFormatter(ch_formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-result_name', type=str, default='test')

    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    # ログファイル名を実験条件に合わせて指定するようにする
    logger = setup_logger(name = __name__, logfile=args.result_name)
    
    main(args)
