from __future__ import division, print_function, absolute_import

import pdb
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

class Learner(nn.Module):

    def __init__(self, image_size, bn_eps, bn_momentum, n_classes):
        super(Learner, self).__init__()
        self.model = nn.ModuleDict(
            {'features': nn.Sequential(
                OrderedDict(
                    [('conv1', nn.Conv2d(3, 32, 3, padding=1)),
                    ('norm1', nn.BatchNorm2d(32, bn_eps, bn_momentum)),
                    ('relu1', nn.ReLU(inplace=False)),
                    ('pool1', nn.MaxPool2d(2)), # half 1 time

                    ('conv2', nn.Conv2d(32, 32, 3, padding=1)),
                    ('norm2', nn.BatchNorm2d(32, bn_eps, bn_momentum)),
                    ('relu2', nn.ReLU(inplace=False)),
                    ('pool2', nn.MaxPool2d(2)), # half 2 time

                    ('conv3', nn.Conv2d(32, 32, 3, padding=1)),
                    ('norm3', nn.BatchNorm2d(32, bn_eps, bn_momentum)),
                    ('relu3', nn.ReLU(inplace=False)),
                    ('pool3', nn.MaxPool2d(2)), # half 3 time

                    ('conv4', nn.Conv2d(32, 32, 3, padding=1)),
                    ('norm4', nn.BatchNorm2d(32, bn_eps, bn_momentum)),
                    ('relu4', nn.ReLU(inplace=False)),
                    ('pool4', nn.MaxPool2d(2)) # half 4 time
                    ]
                )
            )
        })

        clr_in = image_size // 2**4 # feature map size
        self.model.update({'cls': nn.Linear(32 * clr_in * clr_in, n_classes)})
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.model.features(x)
        x = torch.reshape(x, [x.size(0), -1]) # just flatten the feature map
        outputs = self.model.cls(x)
        return outputs # class prediction

    def get_flat_params(self):
        return torch.cat(
            [p.view(-1) for p in self.model.parameters()], 0)

    def copy_flat_params(self, cI):
        idx = 0
        for p in self.model.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen

    def transfer_params(self, learner_w_grad, cI):
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (tensor类型, NOT nn.Parameters anymore).
        # learner_wo_grad 中的参数 用tensor类型 替换了 learner_w_grad 的 parameter类型参数
        # 而view 或者 view_as 是不保留梯度的
        '''
            input = torch.tensor([2., 3.], requires_grad=True)
            output = input**2
            output.sum().backward()
            input.grad
            Out[194]: tensor([4., 6.])
            input_view = input.view(input.shape)
            input_view.grad # 报错,显示input_view不是计算图的叶子结点,没有gradient信息
        
        
        '''
        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen

    def reset_batch_stats(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

