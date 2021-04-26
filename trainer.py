# -*- coding: utf-8 -*-

import torch
import scipy
import numpy as np
from torch import nn
from logging import getLogger
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import models
import utils

logger = getLogger('__main__').getChild('trainer')

def _get_optimizer(optimizer_name):
    if optimizer_name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif optimizer_name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim

def discount(x, amount):
    # scipyによるFIRフィルタによる波形整形
    # https://helve-python.hatenablog.jp/entry/2018/06/18/000000
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]


class Trainer:
    def __init__(self, args, dataset):
        '''Constructor for training algorithm.

        Args:
            args: From command line, picked up by `argparse`.
            dataset: PyTorch Dataloader.
        
        Initializes:
            - Model : child and cotroller.
            - Inference: optimizers for child and controller paramters.
            - Criticism : cross-entropy loss for training the child model.
        '''

        self.args = args
        self.cuda = args.cuda
        torch.cuda.set_device(self.args.cuda_num)
        self.dataset = dataset
        self.epoch = 0
        self.step = 0
        self.train_step = 0
        self.start_epoch = 0

        self.train_data = dataset.train
        self.valid_data = dataset.valid
        self.valid_iter = self.reset_valid_iter()
        self.test_data = dataset.test

        self.epoch_columns = ['epoch', 'train_loss', 'train_acc', 
                            'val_loss', 'val_acc', 'test_loss', 'test_acc']
        self.epoch_results_container = utils.ResultContainer(
                                        self.epoch_columns, 
                                        self.args.model_dir,
                                        'epoch_results_'+utils.get_time()+'.csv')

        self.iter_columns = ['epoch', 'step', 'policy', 'train_loss', 'train_acc', 
                            'val_loss', 'val_acc']
        self.iter_results_container = utils.ResultContainer(
                                        self.iter_columns, 
                                        self.args.model_dir,
                                        'iteration_results_'+utils.get_time()+'.csv')
        
        self.controller_columns = ['epoch', 'step', 'loss', 'reward', 'adv', 'entropy', 'baseline']
        self.controller_results_container = utils.ResultContainer(
                                        self.controller_columns, 
                                        self.args.model_dir,
                                        'controller_results_'+utils.get_time()+'.csv')

        if self.args.use_tensorboard:
            self.tb = SummaryWriter(args.model_dir)
        else:
            self.tb = None
        
        self.baseline = None

        self.shared = self.build_shared_model()
        self.controller = self.build_controller_model()

        shared_optimizer = _get_optimizer(self.args.shared_optim)
        controller_optimizer = _get_optimizer(self.args.controller_optim)
        logger.info('optimzier shared: {}, controller: {}'.
                        format(shared_optimizer, controller_optimizer))

        self.shared_optim = shared_optimizer
        self.controller_optim = controller_optimizer(
            self.controller.parameters(),
            lr=self.args.controller_lr)

        self.ce = nn.CrossEntropyLoss()

    def build_shared_model(self):
        model = models.load_pretrained_network(self.args.pre_trained_network,
                                                     self.args.num_class)
        if self.args.num_gpu == 1:
            model.to('cuda:{}'.format(self.args.cuda_num))
        elif self.args.num_gpu > 1:
            raise NotImplementedError('num_gpu > 1 is in progress')
        return model

    def build_controller_model(self):
        
        model = models.Controller(self.args)

        if self.args.num_gpu == 1:
            model.to('cuda:{}'.format(self.args.cuda_num))
        elif self.args.num_gpu > 1:
            raise NotImplementedError('num_gpu > 1 is in progress')
            
        return model

    def reset_valid_iter(self):
        return self.valid_data.__iter__()


    def train_without_RL(self):
        policy = utils.get_random_policy(self.args.num_blocks, len(self.args.shared_lr_list))
        logger.info('policy:{}'.format(policy))

        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            self.shared.train()

            epoch_train_loss = 0
            epoch_train_corrects = 0

            train_loss = 0
            train_acc = 0
            train_count = 0
            train_correct = 0
            for self.train_step, (inputs, labels) in enumerate(self.train_data):
                if self.args.cuda:
                    inputs = inputs.to('cuda:{}'.format(self.args.cuda_num))
                    labels = labels.to('cuda:{}'.format(self.args.cuda_num))
                
                optimizer = self.set_lr_optimizer(self.shared, self.shared_optim, policy)        
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    loss, corrects = self.get_loss(self.shared, inputs, labels)

                    loss.backward()
                    optimizer.step()

                epoch_train_loss += loss.data
                epoch_train_corrects += corrects.item()
                
                train_loss += loss.data
                train_count += labels.size(0)
                train_correct += corrects.item()

                # 指定したイテレータ分、学習した後に検証データ正解率を求める
                if (self.train_step + 1) % self.args.shared_train_iteration == 0:

                    utils.log_optimizer_lr(optimizer)
                    train_loss_avg = train_loss.item() / self.args.shared_train_iteration
                    train_acc = train_correct / train_count
                    
                    val_acc, val_loss_avg = self.get_validation_acc(self.shared)
                    self._summarize_shared_train(policy, train_loss_avg, train_acc, val_loss_avg, val_acc)

                    policy = utils.get_random_policy(self.args.num_blocks, len(self.args.shared_lr_list))
                    logger.info('policy:{}'.format(policy))

                    train_loss = 0
                    train_acc = 0
                    train_count = 0
                    train_correct = 0
                    self.shared.train()

                self.step += 1
                
            train_epoch_loss = epoch_train_loss.item() / len(self.train_data.dataset)
            train_epoch_acc = epoch_train_corrects / len(self.train_data.dataset)
            val_epoch_loss, val_epoch_acc = self.get_epoch_loss_acc(self.shared, self.valid_data)
            test_epoch_loss, test_epoch_acc = self.get_epoch_loss_acc(self.shared, self.test_data)       
            
            self._summarize_shared_epoch(train_epoch_loss, train_epoch_acc, 
                            val_epoch_loss, val_epoch_acc, test_epoch_loss, test_epoch_acc)

    def train(self):
        
        policy, log_probs, entropies = self.controller.sample(with_details=True)
        logger.info('policy:{}'.format(policy))

        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            self.shared.train()
            self.controller.eval()
            epoch_train_loss = 0
            epoch_train_corrects = 0

            train_loss = 0
            train_acc = 0
            train_count = 0
            train_correct = 0
            for self.train_step, (inputs, labels) in enumerate(self.train_data):
                if self.args.cuda:
                    inputs = inputs.to('cuda:{}'.format(self.args.cuda_num))
                    labels = labels.to('cuda:{}'.format(self.args.cuda_num))
                
                optimizer = self.set_lr_optimizer(self.shared, self.shared_optim, policy)        
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    loss, corrects = self.get_loss(self.shared, inputs, labels)

                    loss.backward()
                    optimizer.step()

                epoch_train_loss += loss.data
                epoch_train_corrects += corrects.item()
                
                train_loss += loss.data
                train_count += labels.size(0)
                train_correct += corrects.item()

                # 指定したイテレータ分、学習した後に検証データ正解率を求める
                if (self.train_step + 1) % self.args.shared_train_iteration == 0:

                    utils.log_optimizer_lr(optimizer)
                    train_loss_avg = train_loss.item() / self.args.shared_train_iteration
                    train_acc = train_correct / train_count
                    
                    val_acc, val_loss_avg = self.get_validation_acc(self.shared)
                    self._summarize_shared_train(policy, train_loss_avg, train_acc, val_loss_avg, val_acc)

                    # TODO(nagae): controllerの学習の調整
                    self.shared.eval()
                    self.train_controller(val_acc, log_probs, entropies)
    
                    policy, log_probs, entropies = self.controller.sample(with_details=True)
                    
                    logger.info('policy:{}'.format(policy))

                    train_loss = 0
                    train_acc = 0
                    train_count = 0
                    train_correct = 0
                    self.shared.train()
                    self.controller.eval()
                self.step += 1
                
            train_epoch_loss = epoch_train_loss.item() / len(self.train_data.dataset)
            train_epoch_acc = epoch_train_corrects / len(self.train_data.dataset)
            val_epoch_loss, val_epoch_acc = self.get_epoch_loss_acc(self.shared, self.valid_data)
            test_epoch_loss, test_epoch_acc = self.get_epoch_loss_acc(self.shared, self.test_data)       
            
            self._summarize_shared_epoch(train_epoch_loss, train_epoch_acc, 
                            val_epoch_loss, val_epoch_acc, test_epoch_loss, test_epoch_acc)

    def get_epoch_loss_acc(self, model, dataloader):
        model.eval()
        total_loss = 0
        total_corrects = 0
        for inputs, labels in tqdm(dataloader):
            if self.args.cuda:
                inputs = inputs.to('cuda:{}'.format(self.args.cuda_num))
                labels = labels.to('cuda:{}'.format(self.args.cuda_num))
            
            with torch.set_grad_enabled(False):
                loss, corrects = self.get_loss(model, inputs, labels)
                   
            total_loss += loss.item() * inputs.size(0) # ミニバッチの合計損失
            total_corrects += corrects.item()

        total_loss = total_loss / len(dataloader.dataset)
        total_acc = total_corrects / len(dataloader.dataset)

        model.train()

        return total_loss, total_acc 
    
    def get_loss(self, model, inputs, labels):
        outputs = model(inputs)
        loss = self.ce(outputs, labels)
        
        _, preds = torch.max(outputs, 1)

        corrects = torch.sum(preds == labels.data)

        return loss, corrects

    # TODO(nagae): InceptionV3以外のネットワークにも対応させる
    def set_lr_optimizer(self, model, optimizer, policy):

        # 全結合層含めて、学習率を設定する        
        params_lr_list = []
        for block_num, (name, module) in enumerate(model.named_children()):
            params_lr_list.append({'params': [p for p in module.parameters()], 
                                    'lr': self.args.shared_lr_list[policy[block_num]]})

        return optimizer(params_lr_list, weight_decay=self.args.shared_l2_reg)

    def get_validation_acc(self, model):
        model.eval()
        
        # validation accuracyの計算
        total_count = 0
        correct_count = 0
        val_loss = 0
        val_step = 0
        logger.debug('validate shared model start step {}'.format(
            self.args.shared_valid_iteration))

        val_step = 0
        while val_step < self.args.shared_valid_iteration:
            
            inputs, labels = next(self.valid_iter)
            if self.args.cuda:
                inputs = inputs.to('cuda:{}'.format(self.args.cuda_num))
                labels = labels.to('cuda:{}'.format(self.args.cuda_num))

            with torch.set_grad_enabled(False):
                loss, corrects = self.get_loss(model, inputs, labels)
            
            correct_count += corrects.item()
            total_count += labels.size(0)
            val_loss += loss

            if self.valid_iter._num_yielded == len(self.valid_data):
                self.valid_iter = self.reset_valid_iter()
                logger.debug('initialize validation iterator')

            val_step += 1

        val_acc = correct_count / total_count

        val_loss_avg = val_loss.item() / val_step
        logger.debug(f'train shared model complete '
                     f'| loss: {val_loss_avg:.3f} | acc {val_acc:.3f}')

        model.train()
        return val_acc, val_loss_avg
        
    def train_controller(self, val_acc, log_probs, entropies):
        self.shared.eval()
        self.controller.train()

        np_entropies = entropies.data.cpu().numpy()
        rewards = val_acc + self.args.entropy_coeff * np_entropies
        
        if 1 > self.args.discount > 0:
            rewards = discount(rewards, self.args.discount) 
        
        # REINFORCE with baseline
        # moving average baseline
        if self.baseline is None:
            if self.args.init_baseline == 'reward':
                self.baseline = rewards
            elif self.args.init_baseline == 'zero':
                self.baseline = 0
            else:
                raise ValueError('{} is not supported for init baseline'.format(self.args.init_baseline))
        else:
            decay = self.args.ema_baseline_decay
            self.baseline = decay * self.baseline + (1 - decay) * rewards

        adv = rewards - self.baseline
        
        # policy loss
        loss = -log_probs * utils.get_variable(adv, 
                                                self.cuda, 
                                                requires_grad=False)
        
        #loss.mean() でも可能、どちらを用いるか要検討
        loss = loss.sum() 

        self.controller_optim.zero_grad()
        loss.backward()

        # 勾配クリッピング, RNNの学習でよく行われるテクニック
        # https://www.madopro.net/entry/rnn_lm_on_wikipedia
        if self.args.controller_grad_clip > 0:
            torch.nn.utils.clip_grad_norm(self.controller.parameters(),
                                            self.args.controller_grad_clip)
        self.controller_optim.step()
        
        np_loss = loss.data.item()

        self._summarize_controller_train(np_loss, adv, rewards, entropies, self.baseline)
        self.controller.eval()               

    def _summarize_shared_epoch(self, train_loss, train_acc, 
                                      val_loss, val_acc, test_loss, test_acc): 

        logger.info(f'| epoch {self.epoch:3d} '
                    f'| train loss {train_loss:.2f} '
                    f'| train acc {train_acc:.3f} '
                    f'| val loss {val_loss:.2f} '
                    f'| val acc {val_acc:.3f} '
                    f'| test loss {test_loss:.2f} '
                    f'| test acc {test_acc:.3f} ')

        if self.tb is not None:
            self.tb.add_scalar('epoch shared/train loss', 
                                train_loss, 
                                self.epoch)

            self.tb.add_scalar('epoch shared/train acc', 
                                train_acc, 
                                self.epoch)

            self.tb.add_scalar('epoch shared/val loss', 
                                val_loss, 
                                self.epoch)

            self.tb.add_scalar('epoch shared/val acc', 
                                val_acc, 
                                self.epoch)
        
            self.tb.add_scalar('epoch shared/test loss', 
                                test_loss, 
                                self.epoch)

            self.tb.add_scalar('epoch shared/test acc', 
                                test_acc, 
                                self.epoch)                              

        self.epoch_results_container.add([self.epoch, train_loss, train_acc, 
                                        val_loss, val_acc, test_loss, test_acc])
        
    def _summarize_shared_train(self, policy, train_loss, train_acc, val_loss, val_acc):

        logger.info(f'| epoch {self.epoch:3d} '
                    f'| step {self.train_step:3d} '
                    f'| train loss {train_loss:.2f} '
                    f'| train acc {train_acc:.3f} '
                    f'| val loss {val_loss:.2f} '
                    f'| val acc {val_acc:.3f} ')
        
        if self.tb is not None:
            self.tb.add_scalar('shared/train loss', 
                                train_loss, 
                                self.step)

            self.tb.add_scalar('shared/train acc', 
                                train_acc, 
                                self.step)

            self.tb.add_scalar('shared/val loss', 
                                val_loss, 
                                self.step)

            self.tb.add_scalar('shared/val acc', 
                                val_acc, 
                                self.step)

        self.iter_results_container.add([self.epoch, self.step, policy, 
                                    train_loss, train_acc, val_loss, val_acc])
        
    def _summarize_controller_train(self,
                                    loss,
                                    adv,
                                    rewards,
                                    entropies,
                                    baseline):
        """Logs the controller's progress for this training epoch."""

        avg_reward = np.mean(rewards)
        avg_adv = np.mean(adv)
        np_entropies = entropies.data.cpu().numpy()
        avg_entropy = np.mean(np_entropies)
        avg_baseline = np.mean(baseline)

        logger.info(
            f'| epoch {self.epoch:3d} ' 
            f'| R {avg_reward:.5f} '
            f'| controller loss {loss:.5f} '
            f'| entropy {avg_entropy:.5f}'
            f'| baseline {avg_baseline:.5f}')
        
        if self.tb is not None:

            self.tb.add_scalar('controller/loss',
                                loss,
                                self.step)

            self.tb.add_scalar('controller/reward',
                                avg_reward,
                                self.step)
                        
            self.tb.add_scalar('controller/adv',
                                avg_adv,
                                self.step)

            self.tb.add_scalar('controller/entropy',
                                avg_entropy,
                                self.step)

            self.tb.add_scalar('controller/avg_baseline',
                                avg_baseline,
                                self.step)  

            # TODO(nagae): policyからのモデルの描画機能の作成

        self.controller_results_container.add([self.epoch, self.step, loss, avg_reward, 
                                                avg_adv, avg_entropy, avg_baseline])

    def draw_cotroller_graph(self):
        def init_hidden(batch_size):
            zeros = torch.zeros(batch_size, self.args.controller_hid)
            return (utils.get_variable(zeros, self.args.cuda, requires_grad=False),
                    utils.get_variable(zeros.clone(), self.args.cuda, requires_grad=False),)
        def _get_default_hidden(key):
            return utils.get_variable(torch.zeros(key, self.args.controller_hid),
                                    self.args.cuda,
                                    requires_grad = False)

        batch_size=1
        static_inputs = utils.KeyDefaultDict(_get_default_hidden)
        static_init_hidden = utils.KeyDefaultDict(init_hidden)
        inputs = static_inputs[batch_size]
        hidden = static_init_hidden[batch_size]
        
        for block_idx in range(self.args.num_blocks):
            dummy_data = (inputs, hidden, torch.Tensor([block_idx]), torch.BoolTensor(1))    
            self.tb.add_graph(self.controller, dummy_data)
