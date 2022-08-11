from __future__ import division, print_function, absolute_import

import os
import pdb
import copy
import random
import argparse
from argparse import ArgumentParser

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from learner import Learner
from metalearner import MetaLearner
from dataloader import prepare_data
from utils import *


FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--mode', choices=['train', 'test'])
# Hyper-parameters
# 每个类别所含样本数
FLAGS.add_argument('--n-shot', type=int,
                   help="How many examples per class for training (k, n_support)")

FLAGS.add_argument('--n-eval', type=int,
                   help="How many examples per class for evaluation (n_query)")
FLAGS.add_argument('--n-class', type=int,
                   help="How many classes (N, n_way)")
FLAGS.add_argument('--input-size', type=int,
                   help="Input size for the first LSTM")
FLAGS.add_argument('--hidden-size', type=int,
                   help="Hidden size for the first LSTM")
FLAGS.add_argument('--lr', type=float,
                   help="Learning rate")
FLAGS.add_argument('--episode', type=int,
                   help="Episodes num to train , means that num of update parameter for meta-learner")
FLAGS.add_argument('--episode-val', type=int,
                   help="Episodes num to eval , num of evaluation meta-learner ")
FLAGS.add_argument('--epoch', type=int,
                   help="Epoch to train for an episode")
FLAGS.add_argument('--batch-size', type=int,
                   help="Batch size when training an episode")
FLAGS.add_argument('--image-size', type=int,
                   help="Resize image to this size")
FLAGS.add_argument('--grad-clip', type=float,
                   help="Clip gradients larger than this number")
FLAGS.add_argument('--bn-momentum', type=float,
                   help="Momentum parameter in BatchNorm2d")
FLAGS.add_argument('--bn-eps', type=float,
                   help="Eps parameter in BatchNorm2d")

# Paths
FLAGS.add_argument('--data', choices=['miniimagenet'],
                   help="Name of dataset")
FLAGS.add_argument('--data-root', type=str,
                   help="Location of data")
FLAGS.add_argument('--resume', type=str,
                   help="Location to pth.tar")
FLAGS.add_argument('--save', type=str, default='logs',
                   help="Location to logs and ckpts")
# Others
# action = 'store_true'的时候,表示只要加上 --cpu 则默认为true
FLAGS.add_argument('--cpu', action='store_true',
                   help="Set this to use CPU, default use CUDA")
FLAGS.add_argument('--n-workers', type=int, default=4,
                   help="How many processes for preprocessing")
FLAGS.add_argument('--pin-mem', type=bool, default=False,
                   help="DataLoader pin_memory")
FLAGS.add_argument('--log-freq', type=int, default=100,
                   help="Logging frequency")
FLAGS.add_argument('--val-freq', type=int, default=1000,
                   help="Validation frequency")
FLAGS.add_argument('--seed', type=int,
                   help="Random seed")


def meta_test(eps, eval_loader, learner_w_grad, learner_wo_grad, metalearner, args, logger):
    for subeps, (episode_x, episode_y) in enumerate(tqdm(eval_loader, ascii=True)):
        # 根据args.episode_val指定的次数,总共从数据中抽args.episode_val次,每次args.num_class个类别的数据
        # 每个类再相应抽 n_shot + n_eval ,其中每个类中的n_shot个图片用作测试集的训练集上的loss
        # 去更新meta-learner
        train_input = episode_x[:, :args.n_shot].reshape(-1, *episode_x.shape[-3:]).to(args.dev) # [n_class * n_shot, :]
        train_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_shot)).to(args.dev) # [n_class * n_shot]
        # 测试集的训练集,用来计算这个special task 上learner的loss,然后计算得到相应
        # 属于这个special task的meta-learner的CI参数,然后最后用CI初始化属于
        # 这个task的learner,然后在这个测试集的测试集上计算loss性能.
        test_input = episode_x[:, args.n_shot:].reshape(-1, *episode_x.shape[-3:]).to(args.dev) # [n_class * n_eval, :]
        test_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_eval)).to(args.dev) # [n_class * n_eval]

        # Train learner with metalearner
        learner_w_grad.reset_batch_stats()
        learner_wo_grad.reset_batch_stats()
        learner_w_grad.train_and_val()
        learner_wo_grad.eval()
        # here parameter of meta-learner  is fixed
        # just used to calculate CI ,that's  parameter of learner of special task
        cI = train_learner(learner_w_grad, metalearner, train_input, train_target, args)

        # Initialize the parameter of learner of special task from meta-learner
        learner_wo_grad.transfer_params(learner_w_grad, cI)
        output = learner_wo_grad(test_input)
        loss = learner_wo_grad.criterion(output, test_target)
        acc = accuracy(output, test_target)
 
        logger.batch_info(loss=loss.item(), acc=acc, phase='eval')

    return logger.batch_info(eps=eps, totaleps=args.episode_val, phase='evaldone')


def train_learner(learner_w_grad, metalearner, train_input, train_target, args):
    cI = metalearner.metalstm.cI.data
    #  Initialize parameter form last meta-learner  for  learner in this loop
    #  每次大循环一次,meta-learner的
    hs = [None]
    # 每次在大循环调用train_learner,都要初始化一下属于每次大循环的数据的初始隐状态
    # 这是因为每个大循环(eposide)用到的数据类别不一样,产生的hidden state不能互相传递
    for _ in range(args.epoch):
        for i in range(0, len(train_input), args.batch_size):
            # batchsize = 25
            x = train_input[i:i+args.batch_size]
            y = train_target[i:i+args.batch_size]

            # get the loss/grad
            learner_w_grad.copy_flat_params(cI)
            # 使用进过meta-learn计算过的CI作为learner的新一次parameter

            output = learner_w_grad(x) # class prediction use learner
            loss = learner_w_grad.criterion(output, y)
            acc = accuracy(output, y)
            learner_w_grad.zero_grad()
            loss.backward() # 注意这里仅仅反向传播计算gradient,并没有执行任何的update parameter过程

            # parameter  gradient  of learner
            grad = torch.cat(
                [p.grad.data.view(-1) / args.batch_size for p in learner_w_grad.parameters()], 0
            )

            # preprocess grad & loss and metalearner forward
            grad_prep = preprocess_grad_loss(grad)
            # [n_learner_params, 2] : 一种梯度处理形式,将这两者都作为输入给meta-learner

            # parameter gradient  of learner
            loss_prep = preprocess_grad_loss(loss.data.unsqueeze(0)) # [1, 2] # 同理

            metalearner_input = [loss_prep, grad_prep, grad.unsqueeze(1)]

            cI, h = metalearner(inputs = metalearner_input, hs = hs[-1])
            # here just used metalearner to update CI that parameter of learner,
            # do not modified parameter of metalearner

            # 返回的 h 包含 2 部分 : [(lstmhx,  lstmcx), metalstm_hs],
            # 前一项是LSTMcell输出的并行隐状态和state ,用于保留参数的上一次信息,
            # 后一项是METALSTMcell的隐状态,用于下一次所有



            hs.append(h)

            print("training loss: {:8.6f} acc: {:6.3f}, mean grad: {:8.6f}".format(loss, acc, torch.mean(grad)))

    return cI


def main():

    args, unparsed = FLAGS.parse_known_args()
    if len(unparsed) != 0:
        raise NameError("Argument {} not recognized".format(unparsed))

    if args.seed is None:
        args.seed = random.randint(0, 1e3)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cpu:
        args.dev = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        args.dev = torch.device('cuda')

    logger = GOATLogger(args)

    # Get meta-data : meta-train-loader ...
    train_loader, val_loader, test_loader = prepare_data(args)
    
    # 初始化 learner, meta-learner , 这时两个model内部的参数已经默认初始化过了
    learner_w_grad = Learner(
        args.image_size, args.bn_eps, args.bn_momentum, args.n_class).to(args.dev)
    # Note : here learner_w_grad just the model of learner
    # it used to store status of parameters , special that's batchnorm values
    learner_wo_grad = copy.deepcopy(learner_w_grad)  # 拷贝对象及其子对象
    # why need deep cpoy :
    #   https://androidkt.com/copy-pytorch-model-using-deepcopy-and-state_dict/
    '''
    The deepcopy will recursively copy every member of an object, 
    so it copies everything. It makes a deep copy of the original tensor 
    meaning it creates a new tensor instance with a new memory 
    allocation to the tensor data.  **The history will not be copied***, 
    as you cannot call copy.deepcopy on a non-leaf tensor.
    '''

    metalearner = MetaLearner(
        args.input_size, args.hidden_size, learner_w_grad.get_flat_params().size(0)
    ).to(args.dev)

    # CI  is the output of lstm meta-learner , and it's also the parameter of learner
    metalearner.metalstm.init_cI(learner_w_grad.get_flat_params())

    # Set up loss, optimizer, learning rate scheduler
    optim = torch.optim.Adam(metalearner.parameters(), args.lr)

    if args.resume:
        logger.loginfo("Initialized from: {}".format(args.resume))
        last_eps, metalearner, optim = resume_ckpt(metalearner, optim, args.resume, args.dev)

    if args.mode == 'test':
        _ = meta_test(last_eps, test_loader, learner_w_grad, learner_wo_grad, metalearner, args, logger)
        return

    best_acc = 0.0
    logger.loginfo("Start training")
    # Meta-training
    for eps, (episode_x, episode_y) in enumerate(train_loader):
        # train_loader 调用 n_class个对应类别的dataloader,相应的dataloader又在自己的类别中抽n_shot+n_eval个image
        # 所以返回 :
        # episode_x.shape = [n_class, n_shot + n_eval, c, h, w]
        # episode_y.shape = [n_class, n_shot + n_eval] --> NEVER USED
        train_input = episode_x[:, :args.n_shot].reshape(-1, *episode_x.shape[-3:]).to(args.dev)
        # [n_class * n_shot, :] , 初始参数为n_class = 5 , n_shot = 5 ,n_eval = 15
        # 表示每次大循环 使用25张image去train learner,然后计算在剩下75张image上的loss,根据loss更新meat-learner

        train_target = torch.LongTensor(
            np.repeat(range(args.n_class), args.n_shot)
        ).to(args.dev)
        # [n_class * n_shot]
        # flatten the label  row by row , so now target is like this
        # np.repeat(range(3),3) : array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        # used to calculate loss , then for  update meta-learner
        # 因为 每个大的episode(外循环)每次只抽5个类,下次大循环又是新的5个类,所以train 的时候直接都用1-5类表示
        test_input = episode_x[:, args.n_shot:].reshape(-1, *episode_x.shape[-3:]).to(args.dev) # [n_class * n_eval, :]
        test_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_eval)).to(args.dev) # [n_class * n_eval]

        # Train learner with metalearner
        '''
        paper : because we do not want to collect mean and standard deviation 
        statistics during meta-testing in a way that allows information to leak
        between different datasets (episodes), being considered.
        we need reset the batch norm statistics 
        
        '''
        learner_w_grad.reset_batch_stats()
        # 每个大循环要把内部learner的batch信息刷新是因为,用之前的batch信息一方面可能导致信息泄露,
        learner_wo_grad.reset_batch_stats()
        # 另一方面这数据也不准,大循环随机抽取的数据所属类别 差别可能比较大

        learner_w_grad.train()
        learner_wo_grad.train()
        # model.train VS model.eval():
        # eval 影响的是一些层,比如dropout,batch等,对于dropout层,
        # train的时候,有p的unit是随机失活的,而eval的时候是全部神经元都用的.
        # 此外batch values是train的时候是随批次变的,而eval的时候用的train的均值.
        # 但是eval仍然会自动梯度的计算,(这会导致一些内存和时间的增加),但是不会反向传播

        # 这时候就需要, with torch.no_gard()代码块,冻结各层,不自动计算局部梯度.
        # 但是一般情况下, eval就够了.反正也不执行loss.backward()

        cI = train_learner(learner_w_grad, metalearner, train_input, train_target, args)
        # 这learner_w_grad在里边也是被meta-learner的输出调整过了参数
        # 每次在进入train_learner的第一句话,就是从meta-learner读取初始CI
        # 然后循环的时候先把初始CI复制给learner_w_grad,这样就实现了,每次大循环
        # 的learner_w_grad初始参数都是meta-learner的初始CI(这个值从来没有变过)
        # 每次大循环变得只是meta-learner内部的参数


        # Train meta-learner with validation loss
        learner_wo_grad.transfer_params(learner_w_grad, cI)
        '''
            将learner更新多轮后的参数赋值给结构完全和learner一样的架构.
            这么做是因为,meta-learner在更新learner ( learner_w_grad) 的过程中是参与了运算的
            如果用learner_w_grad每次的输出去计算loss,然后反向传播,这样会比较复杂,
            就会把train learner过程中的gradient传播回去,需要计算二阶导
            所以在train完learner(即learner_w_grad)后,令learner_wo_grad 调用transfer_params 函数, 
            把learner最新的参数复制过来那这时 , learner_wo_grad内部的参数是不具有历史信息的
            具体可见:https://androidkt.com/copy-pytorch-model-using-deepcopy-and-state_dict/
            
                # a
                # Out[143]: tensor([2., 3.], requires_grad=True)
                # b
                # Out[144]: tensor([6., 4.], requires_grad=True)
                # model.state_dict()
                # Out[145]: 
                # OrderedDict([('weight', tensor([[-0.6795, -0.0612]])),
                #  ('bias', tensor([0.1080]))])
                # y = model(a)**2 + model(b)**2
                # y
                # Out[147]: tensor([19.8124], grad_fn=<AddBackward0>)
                # y.backward()
                # a
                # Out[149]: tensor([2., 3.], requires_grad=True)
                # a.grad
                # Out[150]: tensor([5.9495, 6.1756])
                # b.grad
                # Out[151]: tensor([17.7260,  8.5157])
                # model.weight
                # Out[152]: 
                # Parameter containing:
                # tensor([[-0.6795, -0.0612]], requires_grad=True)
                # model.weight.grad
                # Out[153]: tensor([[-56.3015, -42.3162]])
                # model.bias.grad
                # Out[154]: tensor([-11.2963])
                # import copy
                # model_copy = copy.deepcopy(model)
                # model_copy
                # Out[157]: Linear(in_features=2, out_features=1, bias=True)
                # model_copy.weight.grad
                # 输出为空
            
            因此,这时copy的model是不包含之前learner的历史信息的
            只是起到将meta-learner 连接起来的作用.
            然后又把copy的model中的参数 用最后一次在训练集上的CI 进行初始化,
            这样就通过CI 将测试集的loss和meta-learner连接起来
            接着使用learner_wo_grad去test_input上计算loss,反向传播,
            实现只用测试集loss更新meta-learner
            
        '''

        output = learner_wo_grad(test_input)  # learner_wo_grad 在 testdata上测试
        # 注意这里learner_wo_grad虽然是在"测试",但是属于大循环training meta-learner的
        # 一步,因此这时的参数还是需要计算梯度的,才能后边将loss经过learner_wo_grad传到SLTM
        # 因此这里不设置   learner_wo_grad.eval()
        loss = learner_wo_grad.criterion(output, test_target) # 仅使用测试集上的loss去更新meta-learner
        acc = accuracy(output, test_target)
        
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(metalearner.parameters(), args.grad_clip) # clip the gradient
        optim.step() # update the metalearner parameters

        # logger.batch_info(eps=eps, totaleps=args.episode, loss=loss.item(), acc=acc, phase='train')

        # # Meta-validation
        # if eps % args.val_freq == 0 and eps != 0:
        #     save_ckpt(eps, metalearner, optim, args.save)
        #
        #     acc = meta_test(
        #         eps, val_loader, learner_w_grad, learner_wo_grad,
        #         metalearner, args, logger
        #     )
        #     if acc > best_acc:
        #         best_acc = acc
        #         logger.loginfo("* Best accuracy so far *\n")

    logger.loginfo("Done")


if __name__ == '__main__':
    main()
