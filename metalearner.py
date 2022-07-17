from __future__ import division, print_function, absolute_import

import pdb
import math
import torch
import torch.nn as nn
import numpy as np

class MetaLSTMCell(nn.Module):
    """C_t = f_t * C_{t-1} + i_t * \tilde{C_t}"""
    def __init__(self, input_size, hidden_size, n_learner_params):
        super(MetaLSTMCell, self).__init__()
        """Args:
            input_size (int): cell input size, default = 20
            hidden_size (int): should be 1 for every parameter in learner 
            n_learner_params (int): number of learner's parameters
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_learner_params = n_learner_params
        self.WF = nn.Parameter(torch.Tensor(input_size + 2, hidden_size)) # 加2列是因为loss和gradient进行了处理
        self.WI = nn.Parameter(torch.Tensor(input_size + 2, hidden_size))
        self.cI = nn.Parameter(torch.Tensor(n_learner_params, 1))
        self.bI = nn.Parameter(torch.Tensor(1, hidden_size))
        self.bF = nn.Parameter(torch.Tensor(1, hidden_size))

        self.reset_parameters()
        # just called when Initialize the meta-learner in main function

    def reset_parameters(self):
        for weight in self.parameters():
            nn.init.uniform_(weight, -0.01, 0.01)

        # want initial forget value to be high and input value to be low so that 
        #  model starts with gradient descent
        nn.init.uniform_(self.bF, 4, 6)
        nn.init.uniform_(self.bI, -5, -4)

    def init_cI(self, flat_params):
        self.cI.data.copy_(flat_params.unsqueeze(1))

    def forward(self, inputs, hs = None): # [lstmhx, grad] , [ metalstm_fn, metalstm_in, metalstm_cn]
        """Args:

        # 合并起来作为下一个meta-LSTM的输入
        #  hs[1] : [ metalstm_fn, metalstm_in, metalstm_cn]
            inputs = [x_all, grad] <=> [ lstmhx , grad ]
                #  lstmhx : 上一层LSTMcell的输出(shape:(n_par,input_size))
                            每个参数分别对应input_size个值 ,
                            这input_size个值又是由(processed_grad(2列),processed_loss(2列))得到的

                #   grad  : 参数的梯度(shape:(n_par,1)), 每个参数分别对应一个值

            hs = [f_prev, i_prev, c_prev]:
                # [f_prev, i_prev, c_prev] <=> [ metalstm_fn, metalstm_in, metalstm_cn]

                f (torch.Tensor of size [n_learner_params, 1]): forget gate
                # rely on  parameter , gradient of parameter about loss , loss

                i (torch.Tensor of size [n_learner_params, 1]): input gate
                # where in paper like a learning rate

                c (torch.Tensor of size [n_learner_params, 1]): flattened learner parameters
        """
        x_all, grad = inputs
        #  x_all : 就是lstmhx , 即上一层LSTMcell的输出(shape:(n_par,input_size))

        batch, _ = x_all.size()

        if hs is None:
            f_prev = torch.zeros((batch, self.hidden_size)).to(self.WF.device)
            i_prev = torch.zeros((batch, self.hidden_size)).to(self.WI.device)
            c_prev = self.cI
            hs = [f_prev, i_prev, c_prev]

        f_prev, i_prev, c_prev = hs
        
        # f_t = sigmoid(W_f * [grad_t, loss_t, theta_{t-1}, f_{t-1}] + b_f)
        # 这里 c_prev 就是 theta_{t-1}
        f_next = torch.mm(torch.cat((x_all, c_prev, f_prev), 1), self.WF) + self.bF.expand_as(f_prev)
        # i_t = sigmoid(W_i * [grad_t, loss_t, theta_{t-1}, i_{t-1}] + b_i)
        i_next = torch.mm(torch.cat((x_all, c_prev, i_prev), 1), self.WI) + self.bI.expand_as(i_prev)
        # next cell/params
        c_next = torch.sigmoid(f_next).mul(c_prev) - torch.sigmoid(i_next).mul(grad)

        return c_next, [f_next, i_next, c_next] # hs

    def extra_repr(self):
        s = '{input_size}, {hidden_size}, {n_learner_params}'
        return s.format(**self.__dict__)


class MetaLearner(nn.Module):

    def __init__(self, input_size, hidden_size, n_learner_params):
        super(MetaLearner, self).__init__()
        """Args:
            input_size (int): for the first LSTM layer, default = 4
            hidden_size (int): for the first LSTM layer, default = 20
            n_learner_params (int): number of learner's parameters
        """
        self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.metalstm = MetaLSTMCell(input_size=hidden_size, hidden_size=1, n_learner_params=n_learner_params)

    def forward(self, inputs, hs=None):
        """Args:
            inputs = [loss, grad_prep, grad]
                loss (torch.Tensor of size [1, 2])
                grad_prep (torch.Tensor of size [n_learner_params, 2])
                grad (torch.Tensor of size [n_learner_params])

            hs = [ (lstm_hn, lstm_cn) , [metalstm_fn, metalstm_in, metalstm_cn]]

        """
        loss, grad_prep, grad = inputs # unzip the input
        loss = loss.expand_as(grad_prep)
        inputs = torch.cat((loss, grad_prep), 1) # (处理后的grad(两列),处理后的loss(两列))
        # input for lstm : [n_learner_params, 4]

        # 这里的思路是,假设learner有10000个参数,那么相应的,属于每个参数的更新操作为:
        # 将该属于该参数的loss,gradient作为输入(shape为(1,4),序列长度就是参数个数 ),到LSTM中,
        # 输出结果为:对应该参数的更新值. 这也就是LSTM为什么输入size为4 输出size为1 .
        # 再换句话说,LSTM并不是直接把所有参数吃进去,然后同时输出参数的更新值,而是一个一个输入进去,输出其更新值.
        # ****************************************************************************
        # 注意这里有个很重要的点就是原文中提到的parameter share.
        # 本来的想法是,一个参数分配1个LSTM networw进行专门的预测,也就是说,10000个参数
        # 就建立10000个平行的LSTM,每个LSTM只负责一个参数的预测(那么输入就是 序列长度为1,shape还是(1,4)),
        # 但是这样是不可行的,数量太大,因此才想到把所有参数集中到一个序列中,(seq长度为参数个数,每个参数shape为(1,4))
        # 但是这样的话有个问题,就是learner的参数是独立的,即:
        # 第二个参数输出不应该用到由第一个参数生成的隐状态,而是应该用到第二个参数上一次更新产生的隐状态,
        # 这样的话,就需要每次在更新参数时,用自己之前产生的隐状态,而不是其他参数的,所以每次更新都要记录相应参数的隐状态.
        # 同理,每个参数要用到自己上一次更新的参数,而不是上一个参数的结果.
        # (区分上一次指的是自己的,上一个指的是另一个参数,按序列顺序)

        if hs is None:
            hs = [None, None]

        lstmhx, lstmcx = self.lstm(inputs, hs[0])
        # input: [loss(2列), grad_prep(2列)] , 每行属于一个参数 and [h_t-1 , c_t-1],共num_par行
        # output : 每行分别为learner的参数在LSTMCell的隐状态(即输出)
        # 输出的shape : ( num_par , hiddensize)
        #  lstmhx : 上一层LSTMcell的输出(shape:(n_par,input_size))
        # 每个参数分别对应input_size个值,这input_size个值又是
        # 由(processed_grad(2列), processed_loss(2列))得到的

        # 因为上边的lstm用的是LSTMCell,然后hiddenstate 和 cellstate 用的是同一个
        # 因此对于所有的参数(input),除了参数本身数据不一样以外,LSTMCELL内部 其他参数都是一样的
        '''
        rnn = nn.LSTMCell(2,3)
        input = torch.Tensor([[1,2],[1,2],[1,2]]).reshape(-1,2)
        rnn(input)
            (tensor([[-0.4273,  0.0955,  0.2775],
             [-0.4273,  0.0955,  0.2775],
             [-0.4273,  0.0955,  0.2775]], grad_fn=<MulBackward0>),
            tensor([[-0.6656,  0.4269,  0.5556],
             [-0.6656,  0.4269,  0.5556],
             [-0.6656,  0.4269,  0.5556]], grad_fn=<AddBackward0>))
        这样的结果适合LSTM不一样的,LSTM吃一个序列,他会自动迭代,这时序列中第二个输入的结果
        是在第一个输入产生的结果上 的新结果 ,而 LSTMCell 只是平行的将每个输入放进去
        第一个输入的结果不影响第二个输出
        '''
        flat_learner_unsqzd, metalstm_hs = self.metalstm([lstmhx, grad], hs[1])
        #  lstmhx : 上一层LSTMcell的输出(shape:(n_par,input_size)) : 每个参数分别对应input_size个值
        #   grad  : 参数的梯度(shape:(n_par,1)), 每个参数分别对应一个值
        # 合并起来作为下一个meta-LSTM的输入
        #  hs[1] : [ metalstm_fn, metalstm_in, metalstm_cn]

        return flat_learner_unsqzd.squeeze(), [(lstmhx, lstmcx), metalstm_hs]

