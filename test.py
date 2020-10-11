import os
import random
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
from torchvision import utils 
from tensorboardX import SummaryWriter
import time

from dataset import UnalignedDataset
from model_base import ResNetBlock, Generator, Discriminator
from model_cyclegan import CycleGAN


def save_imgs(imgs, name_imgs, epoch_label):
    img_table_name = '{}_'.format(epoch_label) + name_imgs + '.png'
    save_path = os.path.join('generated_imgs', img_table_name)

    utils.save_image(
        imgs,
        save_path,
        nrow=1,
        normalize=True,
        range=(-1, 1)
    )
        
def test(device, log_dir, gpu_ids, batch_size, lr, beta1, lambda_idt, lambda_A, lambda_B, load_epoch, test_loader):
    model = CycleGAN(device=device, log_dir=log_dir, gpu_ids=gpu_ids)
    model.log_dir = log_dir
    print('load model {}'.format(load_epoch))
    model.load('epoch_' + str(load_epoch))

    if not os.path.exists('generated_imgs'):
        os.makedirs('generated_imgs')

    time_list = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            t1 = time.perf_counter()

            real_A = data['A'].to(device)
            real_B = data['B'].to(device)
            fake_B = model.netG_A(real_A)
            fake_A = model.netG_B(real_B)

            save_imgs(real_A, 'real_A', 'epoch_' + str(load_epoch))
            save_imgs(real_B, 'real_B', 'epoch_' + str(load_epoch))
            save_imgs(fake_B, 'fake_B', 'epoch_' + str(load_epoch))
            save_imgs(fake_A, 'fake_A', 'epoch_' + str(load_epoch))

            t2 = time.perf_counter()
            get_processing_time = t2 - t1
            time_list.append(get_processing_time)

            if batch_idx % 10 == 0:
                print('batch: {} / elapsed_time: {} sec'.format(batch_idx, sum(time_list)))
                time_list = []

        print('done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CycleGAN trainer')

    parser.add_argument(
        'name_dataset', type=str, help='name of your testing image dataset'
    )
    parser.add_argument(
        'load_epoch', type=int, default=0, help='epochs for testing '
    )
    parser.add_argument(
        '--path_log', type=str, default='logs', help='path to dict where log of training details was saved'
    )
    parser.add_argument(
        '--gpu_ids', type=int, nargs='+', default=[0], help='gpu ids'
    )
    parser.add_argument(
        '--num_epoch', type=int, default=400, help='total training epochs'
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
    test_dataset = UnalignedDataset(
        name_dataset=args.name_dataset, 
        load_size=args.load_size, 
        crop_size=args.crop_size, 
        is_train=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1,
        shuffle=True
    )

    # train
    test(device, args.path_log, args.gpu_ids, 1, args.lr, args.beta1, args.lambda_idt, args.lambda_A, args.lambda_B,
         args.load_epoch, test_loader)
