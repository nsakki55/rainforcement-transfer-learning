# -*- coding: utf-8 -*-
import os

import torch

import config
import utils
import data
import trainer
import trainer_batch

def main(args):
    torch.manual_seed(args.random_seed)
    config.log_args(args)
    
    if args.num_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)
    dataset = data.Image(args)

    if args.batch_train:
        logger.info('Batch train')
        trnr = trainer_batch.Trainer(args, dataset)
    else:
        logger.info('Single train')
        trnr = trainer.Trainer(args, dataset)

    if args.mode == 'train':
        utils.save_args(args)
        if args.random_policy:
            logger.info('train without RL')
            trnr.train_without_RL()
        else:
            trnr.train()

if __name__ == '__main__':
    args, unparsed = config.get_args()
    utils.prepare_dirs(args)
    logger = utils.get_logger(log_path=args.log_path, name=__name__, level=args.log_level)
    main(args)