import os
import random
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import time

from dataset import UnalignedDataset
from model_base import ResNetBlock, Generator, Discriminator
from model_cyclegan import CycleGAN


def train(device, log_dir, gpu_ids, batch_size, lr, beta1, lambda_idt, lambda_A, lambda_B, lambda_mask, 
          num_epoch, load_epoch, save_freq, train_loader):
    model = CycleGAN(device=device, log_dir=log_dir, gpu_ids=gpu_ids, lr=lr, beta1=beta1,
                     lambda_idt=lambda_idt, lambda_A=lambda_A, lambda_B=lambda_B, lambda_mask=lambda_mask)

    if load_epoch != 0:
        model.log_dir = log_dir
        print('load model {}'.format(load_epoch))
        model.load('epoch_' + str(load_epoch))

    writer = SummaryWriter(log_dir)

    for epoch in range(num_epoch):
        print('epoch {} started'.format(epoch + 1 + load_epoch))
        t1 = time.perf_counter()

        losses = model.train(train_loader)

        t2 = time.perf_counter()
        get_processing_time = t2 - t1

        print('epoch: {}, elapsed_time: {} sec losses: {}'
              .format(epoch + 1 + load_epoch, get_processing_time, losses))

        writer.add_scalar('loss_G_A', losses[0], epoch + 1 + load_epoch)
        writer.add_scalar('loss_D_A', losses[1], epoch + 1 + load_epoch)
        writer.add_scalar('loss_G_B', losses[2], epoch + 1 + load_epoch)
        writer.add_scalar('loss_D_B', losses[3], epoch + 1 + load_epoch)
        writer.add_scalar('loss_cycle_A', losses[4], epoch + 1 + load_epoch)
        writer.add_scalar('loss_cycle_B', losses[5], epoch + 1 + load_epoch)
        writer.add_scalar('loss_idt_A', losses[6], epoch + 1 + load_epoch)
        writer.add_scalar('loss_idt_B', losses[7], epoch + 1 + load_epoch)
        writer.add_scalar('loss_mask', losses[8], epoch + 1 + load_epoch)
        
        # generate images during training
        with torch.no_grad():
            model.generate_imgs("epoch_" + str(epoch + 1 + load_epoch), batch_size)

        if (epoch + 1 + load_epoch) % save_freq == 0:
            model.save('epoch_%d' % (epoch + 1 + load_epoch))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CycleGAN trainer')

    parser.add_argument(
        'name_dataset', type=str, help='name of your training image dataset'
    )
    parser.add_argument(
        '--path_log', type=str, default='logs', help='path to dict where log of training details will be saved'
    )
    parser.add_argument(
        '--gpu_ids', type=int, nargs='+', default=[0], help='gpu ids'
    )
    parser.add_argument(
        '--num_epoch', type=int, default=400, help='total training epochs'
    )
    parser.add_argument(
        '--load_epoch', type=int, default=0, help='epochs to resume training '
    )
    parser.add_argument(
        '--save_freq', type=int, default=1, help='frequency of saving log'
    )
    parser.add_argument(
        '--load_size', type=int, default=256, help='original image sizes to be loaded'
    )    
    parser.add_argument(
        '--crop_size', type=int, default=256, help='image sizes to be cropped'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch_size for each gpu'
    )
    parser.add_argument(
        '--lr', type=int, default=0.0002, help='learning rate'
    )
    parser.add_argument(
        '--beta1', type=int, default=0.5, help='learning rate'
    )
    parser.add_argument(
        '--lambda_idt', type=int, default=5, help='weights of identity loss'
    )    
    parser.add_argument(
        '--lambda_A', type=int, default=10, help='weights of adversarial loss for A to B'
    )    
    parser.add_argument(
        '--lambda_B', type=int, default=10, help='weights of adversarial loss for B to A'
    )    
    parser.add_argument(
        '--lambda_mask', type=int, default=10, help='weights of Mask loss'
    )

    args = parser.parse_args()

    # random seeds
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    # gpu
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print('device {}'.format(device))

    # dataset
    train_dataset = UnalignedDataset(
        name_dataset=args.name_dataset, 
        load_size=args.load_size, 
        crop_size=args.crop_size, 
        is_train=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        drop_last=True, 
        shuffle=True
    )

    # train
    train(device, args.path_log, args.gpu_ids, args.batch_size, args.lr, args.beta1, 
          args.lambda_idt, args.lambda_A, args.lambda_B, args.lambda_mask,
          args.num_epoch, args.load_epoch, args.save_freq, train_loader)



