# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

from torchvision import models
from logging import getLogger
import utils

logger = getLogger('__main__').getChild('models')

def load_pretrained_network(network_name, class_num):
    if network_name.lower() == 'inceptionv3':
        net = models.inception_v3(pretrained=True, aux_logits=False)
        num_features = net.fc.in_features
        net.fc = nn.Linear(in_features=num_features, out_features=class_num)
    
    elif network_name.lower() == 'vgg16':
        net = models.vgg16(pretrained=True)
        num_features = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(in_features=num_features, out_features=class_num)

    elif network_name == 'resnet18':
        net = models.resnet18(pretrained=True)
        num_features = net.fc.in_features
        net.fc = nn.Linear(in_features=num_features, out_features=class_num)

    net.train()

    return net

class Controller(torch.nn.Module):
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args

        # Controllerの各出力を学習率にする
        self.num_tokens = list()
        for _ in range(self.args.num_blocks):
            self.num_tokens += [len(self.args.shared_lr_list)]
        
        logger.debug('num_tokens: {}'.format(self.num_tokens))

        self.func_names = self.args.shared_lr_list

        num_total_tokens = sum(self.num_tokens)
        
        # controller_hid=100
        # https://pytorch.org/docs/master/generated/torch.nn.Embedding.html
        self.encoder = torch.nn.Embedding(num_total_tokens, self.args.controller_hid)

        # LSTMCellとLSTM https://takoroy-ai.hatenadiary.jp/entry/2018/06/10/203531
        self.lstm = torch.nn.LSTMCell(self.args.controller_hid, self.args.controller_hid)
        
        self.decoders = []
        for idx, size in enumerate(self.num_tokens):
            decoder = torch.nn.Linear(args.controller_hid, size)
            self.decoders.append(decoder)

        # PyTorchでlayerをリストに入れる場合は、ModuleListを用いる
        # https://qiita.com/perrying/items/857df46bb6cdc3047bd8
        self._decoders = torch.nn.ModuleList(self.decoders)

        self.reset_parameters()
        
        self.static_init_hidden = utils.KeyDefaultDict(self.init_hidden)
        
        def _get_default_hidden(key):
            return utils.get_variable(torch.zeros(key, self.args.controller_hid),
                                    self.args.cuda,
                                    requires_grad = False)

        self.static_inputs = utils.KeyDefaultDict(_get_default_hidden)

    def reset_parameters(self):
        init_range = self.args.controller_init_range
        
        for param in self.parameters():
            # PyTorchの初期化方法
            # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
            #torch.nn.init.uniform(param, -init_range, init_range)
            param.data.uniform_(-init_range, init_range)
        
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)
            
    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.args.controller_hid)
        return (utils.get_variable(zeros, self.args.cuda, requires_grad=False),
                utils.get_variable(zeros.clone(), self.args.cuda, requires_grad=False),)

    # block_idxが畳み込み層ブロックの番号に対応する
    def forward(self, inputs, hidden, block_idx, is_embed):
        
        # block1 以外の場合, 前の畳み込みblockのembdedding からの出力
        if not is_embed:
            embed = self.encoder(inputs)
        # block1 の場合
        else:
            embed = inputs

        hx, cx = self.lstm(embed, hidden)
        logits = self.decoders[block_idx](hx)

        # softmax_temperature, default = 5.0 温度付きソフトマックス関数
        # https://qiita.com/nkriskeeic/items/db3b4b5e835e63a7f243
        logits /= self.args.softmax_temperature

        # tanhに定数をかけると早期収束を防げるらしい EANSの論文では定数を2.5としている
        # logit clippingという手法
        # https://arxiv.org/abs/1611.09940
        if self.args.mode == 'train':
            logits = (self.args.tanh_c * torch.tanh(logits))

        return logits, (hx, cx)

    def sample(self, batch_size=1, with_details=False, save_dir=None):
        """controllerの各ブロックから学習率を取得する
        """
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size]

        learning_rates = []
        entropies = []
        log_probs = []
        for block_idx in range(self.args.num_blocks):
            logits, hidden = self.forward(inputs,
                                          hidden,
                                          block_idx,
                                          is_embed=(block_idx == 0))
            
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)

            # エントロピーの計算 https://ja.wikipedia.org/wiki/%E6%83%85%E5%A0%B1%E9%87%8F
            # .meanをとっても良い？
            entropy = -(log_prob * probs).sum(1, keepdim=False)
            #確率の指数をとった多項分布からのサンプリングからactionを決定する, index番号を返す
            #https://pytorch.org/docs/master/generated/torch.multinomial.html#torch.multinomial
            action = probs.multinomial(num_samples=1).data
            
            selected_log_prob = log_prob.gather(1, 
                utils.get_variable(action, requires_grad=False))
        
            entropies.append(entropy)
            log_probs.append(selected_log_prob.squeeze(dim=1))
            #inputs = utils.get_variable(
            #    action[:, 0] + len(self.args.shared_lr_list),
            #    requires_grad=False)
            
            inputs = utils.get_variable(
                action[:, 0],
                requires_grad=False)
            
            learning_rates.append(action[:, 0].item())

        if with_details:
            return learning_rates, torch.cat(log_probs), torch.cat(entropies)

        return learning_rates

    def sample_batch(self, batch_size=1, with_details=False, save_dir=None):
        """controllerの各ブロックから学習率を取得する
        """
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size]

        learning_rates = []
        entropies = []
        log_probs = []
        for block_idx in range(self.args.num_blocks):
            logits, hidden = self.forward(inputs,
                                          hidden,
                                          block_idx,
                                          is_embed=(block_idx == 0))
            
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)

            # エントロピーの計算 https://ja.wikipedia.org/wiki/%E6%83%85%E5%A0%B1%E9%87%8F
            # バッチごと、各ブロックのエントロピーの和をとる
            # .meanをとっても良い？
            entropy = -(log_prob * probs).sum(1, keepdim=False)
            
            #確率の指数をとった多項分布からのサンプリングからactionを決定する, index番号を返す
            #https://pytorch.org/docs/master/generated/torch.multinomial.html#torch.multinomial
            action = probs.multinomial(num_samples=1).data
            
            selected_log_prob = log_prob.gather(1, 
                utils.get_variable(action, requires_grad=False))
        
            entropies.append(entropy)
            log_probs.append(selected_log_prob.squeeze(dim=1))
            
            action = action.squeeze(dim=1)
            
            #inputs = utils.get_variable(
            #    action[:, 0] + len(self.args.shared_lr_list),
            #    requires_grad=False)
            #print(action)
            inputs = utils.get_variable(
                action,
                requires_grad=False)
            
            learning_rates.append(action)
        
        learning_rates = torch.stack(learning_rates).transpose(0, 1)
        
        entropies = torch.stack(entropies)
        entropies = entropies.mean(0)
        log_probs = torch.stack(log_probs)
        log_probs = log_probs.mean(0)

        # バッチごとのpolicyと対応する確率の対数の平均、エントロピーの平均を返す
        if with_details:
            return learning_rates, log_probs, entropies

        return learning_rates