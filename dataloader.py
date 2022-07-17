from __future__ import division, print_function, absolute_import

import os
import re
import pdb
import glob
import pickle

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL.Image as PILI
import numpy as np

from tqdm import tqdm


class EpisodeDataset(data.Dataset):

    def __init__(self, root, phase='train', n_shot = 5, n_eval=15, transform=None):
        """Args:
            root (str): path to data
            phase (str): train, val or test
            n_shot (int): how many examples per class for training (k/n_support)
            n_eval (int): how many examples per class for evaluation
                - n_shot + n_eval = batch_size for data.DataLoader of ClassDataset
            transform (torchvision.transforms): data augmentation
        """
        # e.g get train path
        root = os.path.join(root, phase)

        # all labels name in train
        self.labels = sorted(os.listdir(root))

        # all images path fot per class
        images = [glob.glob(os.path.join(root, label, '*')) for label in self.labels]

        # 列表中的每个元素是一个data.DataLoader,
        # 每次(共n_episode次)外循环(meta-learner)随机抽n_class个类别index
        # 然后从当前列表(episode_loader)抽出对应index的loader,
        # 每个loader 再分别在自己所属的类别中随机抽 batchsize 个样本
        self.episode_loader = [data.DataLoader(
            ClassDataset(images = images[idx], label = idx, transform=transform),
            batch_size= n_shot + n_eval, shuffle=True, num_workers=0) for idx, _ in enumerate(self.labels)]
        # 而且这里列表里边的data.dataloader看起来像是递归的那种, 其实真的就只是在每个大循环被调用一次,
        # 不是说其能够在train learner的时候,自动从大循环固定的10个类别中数据中,每次抽batchsize个图片并把这10个类别中
        # 每个类别的600个图片取遍,不是这样的. 这样写只是说下次取的时候能够自动实现shuffle,num_works等.
        # 实际上,上边的dataloader真就是每次大循环就调用一次,episode_loader负责10个类,ClassDataset的这个dataloader
        # 负责每个类里边随机抽batchsize张image,比如batchsize = n_train(5) + n_eval(5) = 10
        # 那么一整次大循环,真就只使用这10*10 = 100张图片去完成一次大循环:50张去更新learner,剩下50张计算learner的loss
        # 然后去更新大循环的meta-learner. 不要被这里的递归写法误导
    def __getitem__(self, idx):
        return next(iter(self.episode_loader[idx]))
        # 返回的元素是一个对应类别为idx的dataloader
        # 而这个dataloader又返回包含idx类别下的图片
        # 注意这个地方在外边的一个大循环内部,因此一个大循环内部的类别idx是不变的

    def __len__(self):
        return len(self.labels)


class ClassDataset(data.Dataset):

    def __init__(self, images, label, transform=None):
        """Args:
            images (list of str): each item is a path to an image of the same label
            label (int): the label of all the images
        """
        self.images = images # images path
        self.label = label
        self.transform = transform

    def __getitem__(self, idx):
        image = PILI.open(self.images[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, self.label

    def __len__(self):
        return len(self.images)


class EpisodicSampler(data.Sampler):

    def __init__(self, total_classes, n_class, n_episode):
        self.total_classes = total_classes
        self.n_class = n_class
        self.n_episode = n_episode

    def __iter__(self):
        for i in range(self.n_episode):
            yield torch.randperm(self.total_classes)[:self.n_class]
            # 假设 n_class = 10 , 表示在内循环训练 learner 时 ,要用到10个类别的样本(每个类600个样本)
            # 共6000个 . 然后再划分用于learner的训练集(比如5000)和测试集(1000), 即每次在外层meta-learner
            # 参数固定的情况下,更新内层循环learner要用到5000个样本,最后再用learner在1000个测试样本上
            # 计算loss , 将loss返回 , 进而更新外层的meta-learner的参数




    def __len__(self):
        return self.n_episode


def prepare_data(args):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 从train文件夹内读取图片
    # EpisodeDataset继承于data.Dataset，是一个dataloader，负责读取不同类别
    # 每次返回的一个元素也是dataloader，这个loader负责读取相应类别内的图片
    # 也就是说train_set 这时是一个dataloader , 一个专门返回meta-train的数据
    train_set = EpisodeDataset(args.data_root, 'train', args.n_shot, args.n_eval,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize]))

    val_set = EpisodeDataset(args.data_root, 'val', args.n_shot, args.n_eval,
        transform=transforms.Compose([
            transforms.Resize(args.image_size * 8 // 7),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize]))

    test_set = EpisodeDataset(args.data_root, 'test', args.n_shot, args.n_eval,
        transform=transforms.Compose([
            transforms.Resize(args.image_size * 8 // 7),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize]))

    train_loader = data.DataLoader(train_set, num_workers=args.n_workers, pin_memory=args.pin_mem,
        batch_sampler = EpisodicSampler(len(train_set), args.n_class, args.episode))
    # 假设n_class = 5 , 则表示每次抽5个类别的dataloader ,
    # 而这个 dataloader 又以 batch 的形式返回对应类别内
    # 的图片(比如batchsize = 5+15) , 因此 实际上train_loader 每次返回的是 ( 5 , 20 , C , H , W )
    # EpisodicSampler 每次返回n_class 个随机索引,也就是每个大循环的所用的类别索引
    val_loader = data.DataLoader(val_set, num_workers=2, pin_memory=False,
        batch_sampler= EpisodicSampler(len(val_set), args.n_class, args.episode_val))


    test_loader = data.DataLoader(test_set, num_workers=2, pin_memory=False,
        batch_sampler = EpisodicSampler(len(test_set), args.n_class, args.episode_val))

    return train_loader, val_loader, test_loader
