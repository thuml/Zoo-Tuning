import argparse
import os
from time import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from module.backbone import build_model
from utils.transforms import get_transforms
from utils.tools import AccuracyMeter, TenCropsTest
from utils.coco70 import COCO70

def get_configs():
    parser = argparse.ArgumentParser(
        description='Pytorch Zoo-Tuning Training')

    # train
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU num for training')
    parser.add_argument('--seed', type=int, default=2020)

    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--total_iter', default=15050, type=int)
    parser.add_argument('--eval_iter', default=1000, type=int)
    parser.add_argument('--test_iter', default=15000, type=int)
    parser.add_argument('--save_iter', default=15000, type=int)
    parser.add_argument('--print_iter', default=100, type=int)

    # dataset
    parser.add_argument('--dataset', default='cifar',
                        type=str, help='Name of dataset')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Num of workers used in dataloading')

    # optimizer
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='Learning rate for training')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma value for learning rate decay')
    parser.add_argument('--nesterov', default=True,
                        type=bool, help='nesterov momentum')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optimizer')
    parser.add_argument('--weight_decay', default=5e-4,
                        type=float, help='Weight decay value for optimizer')
    parser.add_argument('--ratio', default=1.0, type=float)

    # experiment
    parser.add_argument('--lite', default=0, type=int)
    parser.add_argument('--name', default=None, type=str,
                        help='Name of the experiment')
    parser.add_argument('--save_dir', default="model",
                        type=str, help='Path of saved models')
    parser.add_argument('--visual_dir', default="visual",
                        type=str, help='Path of tensorboard data for training')

    configs = parser.parse_args()

    return configs


def str2list(v):
    return v.split(',')


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_writer(log_dir):
    return SummaryWriter(log_dir)


def get_data_loader(configs):
    dataset_dict = {
        'air': ['/data3/zoo-tuning/FGVCAircraft', 100],
        'coco': ['/data3/zoo-tuning/COCO70', 70],
        'cars': ['/data3/zoo-tuning/stanford_cars', 196],
        'mit': ['/data3/zoo-tuning/MITindoors67', 67],
    }
    # data augmentation
    data_transforms = get_transforms(resize_size=256, crop_size=224)

    # build dataset
    if configs.dataset == 'cifar':
        train_dataset = datasets.CIFAR100(root='/data3/zoo-tuning', train=True, transform=data_transforms['train'], download=True)
        val_dataset = datasets.CIFAR100(root='/data3/zoo-tuning', train=False, transform=data_transforms['val'])
        test_datasets = {
            'test' + str(i):
                datasets.CIFAR100(root='/data3/zoo-tuning', train=False, transform=data_transforms['test' + str(i)])
            for i in range(10)
        }
        class_num = 100
    elif configs.dataset == 'coco':
        train_dataset = COCO70(
            dataset_dict[configs.dataset][0], split='train',
            transform=data_transforms['train'])
        val_dataset = COCO70(
            dataset_dict[configs.dataset][0], split='test',
            transform=data_transforms['val'])
        test_datasets = {
            'test' + str(i):
                COCO70(
                    dataset_dict[configs.dataset][0], split='test',
                    transform=data_transforms['test' + str(i)]
                )
            for i in range(10)
        }

        class_num = dataset_dict[configs.dataset][1]
    else:
        train_dataset = datasets.ImageFolder(
                os.path.join(dataset_dict[configs.dataset][0], 'train'),
                transform=data_transforms['train'])
        val_dataset = datasets.ImageFolder(
                os.path.join(dataset_dict[configs.dataset][0], 'test'),
                transform=data_transforms['val'])
        test_datasets = {
            'test' + str(i):
                datasets.ImageFolder(
                    os.path.join(dataset_dict[configs.dataset][0], 'test'),
                    transform=data_transforms['test' + str(i)]
            )
            for i in range(10)
        }

        class_num = dataset_dict[configs.dataset][1]

    # build dataloader
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True,
                              num_workers=configs.num_workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                            num_workers=configs.num_workers, pin_memory=True)

    test_loaders = {
        'test' + str(i):
            DataLoader(
                test_datasets["test" + str(i)],
                batch_size=4, shuffle=False, num_workers=configs.num_workers
        )
        for i in range(10)
    }

    return train_loader, val_loader, test_loaders, class_num


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(configs, train_loader, val_loader, test_loaders, net):
    train_len = len(train_loader) - 1

    # different learning rates for different layers
    params_list = [{"params": filter(lambda p: p.requires_grad, net.f_net.parameters()), "lr": configs.lr},
                   {"params": filter(lambda p: p.requires_grad, net.c_net.parameters()), "lr": configs.lr * configs.ratio}]

    # optimizer and scheduler
    optimizer = torch.optim.SGD(params_list, lr=configs.lr, weight_decay=configs.weight_decay,
                                momentum=configs.momentum, nesterov=configs.nesterov)
                   
    milestones = [6000, 12000, 18000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones, gamma=configs.gamma)

    # check visual path
    visual_path = os.path.join(configs.visual_dir, configs.name)
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    writer = get_writer(visual_path)

    # check model save path
    save_path = os.path.join(configs.save_dir, configs.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # start training
    train_acc_meter = AccuracyMeter(topk=(1,))

    for iter_num in range(configs.total_iter):
        net.train()

        if iter_num % train_len == 0:
            train_iter = iter(train_loader)
        # Data Stage
        data_start = time()

        train_inputs, train_labels = next(train_iter)
        train_inputs, train_labels = train_inputs.cuda(), train_labels.cuda()

        data_duration = time() - data_start

        # Calc Stage
        calc_start = time()

        train_outputs = net(train_inputs)
        
        train_acc_meter.update(train_outputs, train_labels)
        loss = classifier_loss = nn.CrossEntropyLoss()(train_outputs, train_labels) 

        writer.add_scalar('loss/classifier_loss', classifier_loss, iter_num)
        writer.add_scalar('loss/loss', loss, iter_num)

        net.zero_grad()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        scheduler.step()

        calc_duration = time() - calc_start

        if iter_num % configs.eval_iter == 0 and iter_num > 0:

            with torch.no_grad():
                eval_acc_meter = AccuracyMeter(topk=(1,))
                net.eval()
                for val_inputs, val_labels in tqdm(val_loader):
                    val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()
                    val_outputs = net(val_inputs)
                    eval_acc_meter.update(val_outputs, val_labels)

            writer.add_scalar('acc/train_acc', train_acc_meter.avg[1], iter_num)
            writer.add_scalar('acc/val_acc', eval_acc_meter.avg[1], iter_num)
            print('Iter: {} train_acc: {} val_acc: {}'.format(iter_num, train_acc_meter.avg[1], eval_acc_meter.avg[1]))

            train_acc_meter.reset()
            eval_acc_meter.reset()

        if iter_num % configs.test_iter == 0 and iter_num > 0:
            test_acc = TenCropsTest(test_loaders, net)
            
            writer.add_scalar('acc/test_acc', test_acc, iter_num)
            print('Iter: {} test_acc: {}'.format(iter_num, test_acc))

        if iter_num % configs.print_iter == 0:
            print(
                "Iter: {}/{} Loss: {:2f}, d/c: {}/{}".format(iter_num, configs.total_iter, loss, data_duration, calc_duration))


def main():
    configs = get_configs()
    print(configs)
    torch.cuda.set_device(configs.gpu)
    set_seeds(configs.seed)

    train_loader, val_loader, test_loaders, class_num = get_data_loader(configs)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.f_net = build_model(lite=configs.lite)
            self.c_net = nn.Linear(2048, class_num)
            self.c_net.weight.data.normal_(0, 0.01)
            self.c_net.bias.data.fill_(0.0)
                
        def forward(self, x):
            feature = self.f_net(x)
            out = self.c_net(feature)

            return out

    net = Net().cuda()

    train(configs, train_loader, val_loader, test_loaders, net)


if __name__ == '__main__':
    print("PyTorch {}".format(torch.__version__))
    print("TorchVision {}".format(torchvision.__version__))
    main()
