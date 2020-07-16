import os
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import time
import cv2

##################################################################
from dataset import UnalignedDataset
from model_base import ResNetBlock, Generator, Discriminator
from model_cyclegan import CycleGAN
##################################################################


def train(log_dir, device, lr, beta1, lambda_idt, lambda_A, lambda_B, lambda_mask,
          num_epoch, num_epoch_resume, save_epoch_freq):
    model = CycleGAN(log_dir=log_dir, device=device, lr=lr, beta1=beta1,
                     lambda_idt=lambda_idt, lambda_A=lambda_A, lambda_B=lambda_B, lambda_mask=lambda_mask)

    if num_epoch_resume != 0:
        model.log_dir = log_dir
        print('load model {}'.format(num_epoch_resume))
        model.load('epoch' + str(num_epoch_resume))

    writer = SummaryWriter(log_dir)

    for epoch in range(num_epoch):
        print('epoch {} started'.format(epoch + 1 + num_epoch_resume))
        t1 = time.perf_counter()

        losses = model.train(train_loader)

        t2 = time.perf_counter()
        get_processing_time = t2 - t1

        print('epoch: {}, elapsed_time: {} sec losses: {}'
              .format(epoch + 1 + num_epoch_resume, get_processing_time, losses))

        writer.add_scalar('loss_G_A', losses[0], epoch + 1 + num_epoch_resume)
        writer.add_scalar('loss_D_A', losses[1], epoch + 1 + num_epoch_resume)
        writer.add_scalar('loss_G_B', losses[2], epoch + 1 + num_epoch_resume)
        writer.add_scalar('loss_D_B', losses[3], epoch + 1 + num_epoch_resume)
        writer.add_scalar('loss_cycle_A', losses[4], epoch + 1 + num_epoch_resume)
        writer.add_scalar('loss_cycle_B', losses[5], epoch + 1 + num_epoch_resume)
        writer.add_scalar('loss_idt_A', losses[6], epoch + 1 + num_epoch_resume)
        writer.add_scalar('loss_idt_B', losses[7], epoch + 1 + num_epoch_resume)
        writer.add_scalar('loss_mask', losses[8], epoch + 1 + num_epoch_resume)

        if (epoch + 1 + num_epoch_resume) % save_epoch_freq == 0:
            model.save('epoch%d' % (epoch + 1 + num_epoch_resume))


if __name__ == '__main__':

    # random seeds
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    # image
    height = 128
    width = 256

    # training details
    batch_size = 2
    lr = 0.0002  # initial learning rate for adam
    beta1 = 0.5  # momentum term of adam

    num_epoch = 100
    num_epoch_resume = 0
    save_epoch_freq = 1

    # weights of loss function
    # lambda_idt = 0.5
    # lambda_A = 10.0
    # lambda_B = 10.0
    # lambda_mask = 10.0
    lambda_idt = 5.0
    lambda_A = 10.0
    lambda_B = 10.0
    lambda_mask = 10.0

    # files, dirs
    log_dir = 'logs_mask'

    # gpu
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    print('device {}'.format(device))

    # dataset
    train_dataset = UnalignedDataset(is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # train
    train(log_dir, device, lr, beta1, lambda_idt, lambda_A, lambda_B, lambda_mask,
          num_epoch, num_epoch_resume, save_epoch_freq)



