from __future__ import division, print_function, absolute_import

import os
import pdb
import copy
import random
import argparse

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
        train_input = episode_x[:, :args.n_shot].reshape(-1, *episode_x.shape[-3:]).to(args.dev) # [n_class * n_shot, :]
        train_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_shot)).to(args.dev) # [n_class * n_shot]
        test_input = episode_x[:, args.n_shot:].reshape(-1, *episode_x.shape[-3:]).to(args.dev) # [n_class * n_eval, :]
        test_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_eval)).to(args.dev) # [n_class * n_eval]

        # Train learner with metalearner
        learner_w_grad.reset_batch_stats()
        learner_wo_grad.reset_batch_stats()
        learner_w_grad.train()
        learner_wo_grad.eval()
        # get  parameter of learner of special task
        cI = train_learner(learner_w_grad, metalearner, train_input, train_target, args)

        # Initialize the parameter of learner of special task from meta-learner
        learner_wo_grad.transfer_params(learner_w_grad, cI)
        output = learner_wo_grad(test_input)
        loss = learner_wo_grad.criterion(output, test_target)
        acc = accuracy(output, test_target)
 
        logger.batch_info(loss=loss.item(), acc=acc, phase='eval')

    return logger.batch_info(eps=eps, totaleps=args.episode_val, phase='evaldone')


def train_learner(learner_w_grad, metalearner, train_input, train_target, args):
    cI = metalearner.metalstm.cI.data  #  Initialize for  learner in this loop
    hs = [None]
    for _ in range(args.epoch):
        for i in range(0, len(train_input), args.batch_size):
            # batchsize = 25
            x = train_input[i:i+args.batch_size]
            y = train_target[i:i+args.batch_size]

            # get the loss/grad
            learner_w_grad.copy_flat_params(cI)
            # 使用进过meta-learn计算过的CI作为learner的新一次parameter

            output = learner_w_grad(x) # class prediction
            loss = learner_w_grad.criterion(output, y)
            acc = accuracy(output, y)
            learner_w_grad.zero_grad() #
            loss.backward()

            # parameter gradient  of learner
            grad = torch.cat(
                [p.grad.data.view(-1) / args.batch_size for p in learner_w_grad.parameters()], 0
            )

            # preprocess grad & loss and metalearner forward
            grad_prep = preprocess_grad_loss(grad)
            # [n_learner_params, 2] : 一种梯度处理形式,将这两者都作为输入给meta-learner

            # parameter gradient  of learner
            loss_prep = preprocess_grad_loss(loss.data.unsqueeze(0)) # [1, 2] # 同理
            metalearner_input = [loss_prep, grad_prep, grad.unsqueeze(1)] #

            cI, h = metalearner(inputs = metalearner_input, hs = hs[-1])
            # hs[-1] : [f_prev, i_prev, c_prev] -> meta-learner的上一次相关输出
            hs.append(h)

            #print("training loss: {:8.6f} acc: {:6.3f}, mean grad: {:8.6f}".format(loss, acc, torch.mean(grad)))

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
    learner_w_grad = Learner(args.image_size, args.bn_eps, args.bn_momentum, args.n_class).to(args.dev)
    # Note : here learner_w_grad just the model of learner
    # it used to store status of parameters , special that's batchnorm values
    learner_wo_grad = copy.deepcopy(learner_w_grad)  # 拷贝对象及其子对象
    # why need deep cpoy : https://androidkt.com/copy-pytorch-model-using-deepcopy-and-state_dict/
    '''
    The deepcopy will recursively copy every member of an object, 
    so it copies everything. It makes a deep copy of the original tensor 
    meaning it creates a new tensor instance with a new memory 
    allocation to the tensor data.  **The history will not be copied***, 
    as you cannot call copy.deepcopy on a non-leaf tensor.
    '''

    metalearner = MetaLearner(args.input_size, args.hidden_size, learner_w_grad.get_flat_params().size(0)).to(args.dev)

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

        print(eps, episode_x.shape, train_input.shape)

        train_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_shot)).to(args.dev)
        # [n_class * n_shot]
        # flatten the label  row by row , so now target is like this
        # np.repeat(range(3),3) : array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        # used to calculate loss , then for  update meta-learner
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
        learner_wo_grad.reset_batch_stats()

        learner_w_grad.train()
        learner_wo_grad.train()
        # model.train VS model.eval():
        # eval 影响的是一些层,比如dropout,batch等,对于dropout层,
        # train的时候,有p的unit是随机失活的,而eval的时候是全部神经元都用的.
        # 此外batch values是train的时候是随批次变的,而eval的时候用的train的均值.
        # 但是eval不影响自动梯度的计算,(这会导致一些内存和时间的增加),这时候就需要
        # with torch.no_gard()代码块,冻结各层,不自动计算局部梯度.
        # 但是一般情况下, eval就够了.反正也不执行loss.backward()

        cI = train_learner(learner_w_grad, metalearner, train_input, train_target, args)

        # Train meta-learner with validation loss
        learner_wo_grad.transfer_params(learner_w_grad, cI)
        # 将learner更新多轮后的参数赋值给结构完全和learner一样的架构
        # 这么做是因为,meta-learner在更新learner ( learner_w_grad) 的过程中是参与了运算的
        # 如果用learner_w_grad的输出去计算loss,然后反向传播

        output = learner_wo_grad(test_input)
        loss = learner_wo_grad.criterion(output, test_target)
        acc = accuracy(output, test_target)
        
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(metalearner.parameters(), args.grad_clip) # clip the gradient
        optim.step()

        logger.batch_info(eps=eps, totaleps=args.episode, loss=loss.item(), acc=acc, phase='train')

        # Meta-validation
        if eps % args.val_freq == 0 and eps != 0:
            save_ckpt(eps, metalearner, optim, args.save)
            acc = meta_test(eps, val_loader, learner_w_grad, learner_wo_grad, metalearner, args, logger)
            if acc > best_acc:
                best_acc = acc
                logger.loginfo("* Best accuracy so far *\n")

    logger.loginfo("Done")


if __name__ == '__main__':
    main()
