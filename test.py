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


def test(log_dir, device, lr, beta1, lambda_idt, lambda_A, lambda_B, lambda_mask,
          num_epoch, num_epoch_resume, save_epoch_freq, test_loader, epoch_label):
    model = CycleGAN(log_dir=log_dir, device=device, lr=lr, beta1=beta1, lambda_idt=lambda_idt,
                     lambda_A=lambda_A, lambda_B=lambda_B, lambda_mask=lambda_mask, mode_train=False)
    model.log_dir = log_dir
    model.load(epoch_label)

    time_list = []

    for batch_idx, data in enumerate(test_loader):
        t1 = time.perf_counter()

        # generate images
        fake_B = model.netG_A(data['A'].to(device))

        # transpose axis
        real_A = data['A'].permute(0, 2, 3, 1)
        fake_B = fake_B.data.permute(0, 2, 3, 1)

        # [-1,1] => [0, 1]
        real_A = 0.5 * (real_A + 1) * 255
        fake_B = 0.5 * (fake_B + 1) * 255

        # tensor to array
        device2 = torch.device('cpu')
        real_A = real_A.to(device2)
        real_A = real_A.detach().clone().numpy()
        fake_B = fake_B.to(device2)
        fake_B = fake_B.detach().clone().numpy()

        if not os.path.exists('./{}/real_A'.format(log_dir)):
            os.mkdir('./{}/real_A'.format(log_dir))
        if not os.path.exists('./{}/fake_B'.format(log_dir)):
            os.mkdir('./{}/fake_B'.format(log_dir))

        for i in range(real_A.shape[0]):
            file_name = data['path_A'][i].split('/')[3]

            print(file_name)

            save_path_real_A = './{}/real_A/'.format(log_dir) + file_name
            save_path_fake_B = './{}/fake_B/'.format(log_dir) + file_name

            real_A_id_i = real_A[i]
            fake_B_id_i = fake_B[i]

            real_A_id_i = cv2.cvtColor(real_A_id_i, cv2.COLOR_RGB2BGR)
            fake_B_id_i = cv2.cvtColor(fake_B_id_i, cv2.COLOR_RGB2BGR)

            cv2.imwrite(save_path_real_A, real_A_id_i)
            cv2.imwrite(save_path_fake_B, fake_B_id_i)

        t2 = time.perf_counter()
        get_processing_time = t2 - t1
        time_list.append(get_processing_time)

        if batch_idx % 10 == 0:
            print('batch: {} / elapsed_time: {} sec'.format(batch_idx, sum(time_list)))
            time_list = []


if __name__ == '__main__':

    # random seeds
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    # image
    height = 128
    width = 256

    # training details
    batch_size = 1
    lr = 0.0002  # initial learning rate for adam
    beta1 = 0.5  # momentum term of adam

    num_epoch = 100
    num_epoch_resume = 0
    save_epoch_freq = 5

    # weights of loss function
    lambda_idt = 5
    lambda_A = 10.0
    lambda_B = 10.0
    lambda_mask = 0.0

    # files, dirs
    log_dir = 'logs_B8_E50_5_10_10'

    # gpu
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    print('device {}'.format(device))

    # dataset
    test_dataset = UnalignedDataset(is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)

    # test
    epoch_label = 'epoch50'

    test(log_dir, device, lr, beta1, lambda_idt, lambda_A, lambda_B, lambda_mask,
          num_epoch, num_epoch_resume, save_epoch_freq, test_loader, epoch_label)
