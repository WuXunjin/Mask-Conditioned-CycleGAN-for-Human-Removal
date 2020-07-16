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

from dataset import LSTMDataset


class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True),
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3),
                       nn.InstanceNorm2d(dim)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),

            nn.Conv2d(3, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),

            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            # ResNetBlock(256),
            # ResNetBlock(256),
            # ResNetBlock(256),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )

        # initialize weights
        self.model.apply(self._init_weights)

    def forward(self, input_img):
        out = self.model(input_img)
        return out

    @staticmethod
    def _init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal(m.weight.data, 0.0, 0.02)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

        # initialize weights
        self.model.apply(self._init_weights)

    def forward(self, input_img):
        out = self.model(input_img)
        return out

    @staticmethod
    def _init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal(m.weight.data, 0.0, 0.02)


class Reshape(nn.Module):
    def __init__(self, args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class LSTMGenerator_A(nn.Module):
    def __init__(self, batch_size, window_size, step_size, device):
        super(LSTMGenerator_A, self).__init__()
        self.model_1 = Reshape([int(batch_size * window_size / step_size), 3, 128, 256])

        self.model_2 = nn.Sequential(
            nn.ReflectionPad2d(3),

            nn.Conv2d(3, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),

            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            # ResNetBlock(256),
            # ResNetBlock(256),
            # ResNetBlock(256),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )

        self.model_3 = Reshape([int(batch_size), int(window_size / step_size), 3, 128, 256])

        # initialize weights
        self.model_2.apply(self._init_weights)

    def forward(self, input_img):
        out = self.model_1(input_img)
        out = self.model(out)
        out = self.model(out)
        return out

    @staticmethod
    def _init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal(m.weight.data, 0.0, 0.02)


class LSTMGenerator_B(nn.Module):
    def __init__(self, batch_size, window_size, step_size, device):
        super(LSTMGenerator_B, self).__init__()
        self.model_1 = Reshape([int(batch_size * window_size / step_size), 3, 128, 256])

        self.model_2 = nn.Sequential(
            nn.ReflectionPad2d(3),

            nn.Conv2d(3, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),

            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            # ResNetBlock(256),
            # ResNetBlock(256),
            # ResNetBlock(256),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )

        self.model_3 = Reshape([int(batch_size), int(window_size / step_size), 3, 128, 256])

        # initialize weights
        self.model_2.apply(self._init_weights)

    def forward(self, input_img):
        out = self.model_1(input_img.to('cuda:1'))
        out = self.model(out)
        out = self.model_3(out)
        return out.to('cuda:0')

    @staticmethod
    def _init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal(m.weight.data, 0.0, 0.02)


class C3dResNetBlock(nn.Module):
    def __init__(self, dim):
        super(C3dResNetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReplicationPad3d(1),
                       nn.Conv3d(dim, dim, kernel_size=3),
                       nn.InstanceNorm3d(dim),
                       nn.ReLU(True),
                       nn.ReplicationPad3d(1),
                       nn.Conv3d(dim, dim, kernel_size=3),
                       nn.InstanceNorm3d(dim)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Conv3dGenerator(nn.Module):
    def __init__(self):
        super(Conv3dGenerator, self).__init__()

        self.model = nn.Sequential(
            nn.ReplicationPad3d(1),

            nn.Conv3d(3, 64, kernel_size=3),
            nn.InstanceNorm3d(64),
            nn.ReLU(True),

            nn.Conv3d(64, 128, kernel_size=3, stride=[1, 2, 2], padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(True),

            nn.Conv3d(128, 256, kernel_size=3, stride=[1, 2, 2], padding=1),
            nn.InstanceNorm3d(256),
            nn.ReLU(True),

            C3dResNetBlock(256),
            C3dResNetBlock(256),
            C3dResNetBlock(256),
            C3dResNetBlock(256),
            C3dResNetBlock(256),
            C3dResNetBlock(256),
            # ResNetBlock(256),
            # ResNetBlock(256),
            # ResNetBlock(256),

            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=[1, 2, 2], padding=1, output_padding=[0, 1, 1]),
            nn.InstanceNorm3d(128),
            nn.ReLU(True),

            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=[1, 2, 2], padding=1, output_padding=[0, 1, 1]),
            nn.InstanceNorm3d(64),
            nn.ReLU(True),

            nn.ReplicationPad3d(1),
            nn.Conv3d(64, 3, kernel_size=3, stride=1, padding=0),
            nn.Tanh()
        )

        # initialize weights
        self.model.apply(self._init_weights)

    def forward(self, input_img):
        out = input_img.permute([0, 2, 1, 3, 4])
        out = self.model(out)
        out = out.permute([0, 2, 1, 3, 4])
        return out

    @staticmethod
    def _init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal(m.weight.data, 0.0, 0.02)


class Conv3dDiscriminator(nn.Module):
    def __init__(self):
        super(Conv3dDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=[1, 2, 2], padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(64, 128, kernel_size=3, stride=[1, 2, 2], padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(128, 256, kernel_size=3, stride=[1, 2, 2], padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(512, 1, kernel_size=3, stride=1, padding=1)
        )

        # initialize weights
        self.model.apply(self._init_weights)

    def forward(self, input_img):
        out = input_img.permute([0, 2, 1, 3, 4])
        out = self.model(out)
        out = out.permute([0, 2, 1, 3, 4])
        return out.squeeze()

    @staticmethod
    def _init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal(m.weight.data, 0.0, 0.02)


class FrameDiscriminator(nn.Module):

    def __init__(self, batch_size, window_size, step_size):
        super(FrameDiscriminator, self).__init__()

        self.model_1 = Reshape([-1, 3, 128, 256])

        self.model_2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

        # initialize weights
        self.model_2.apply(self._init_weights)

    def forward(self, input_img):
        out = self.model_1(input_img)
        out = self.model_2(out)
        return out.squeeze()

    @staticmethod
    def _init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal(m.weight.data, 0.0, 0.02)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    print('device {}'.format(device))

    batch_size = 2
    window_size = 48
    step_size = 12
    train_dataset = LSTMDataset(batch_size, window_size, step_size, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#
    data = iter(train_loader).next()
    print('===========================================================')
    # print(data['A'].shape)
    # print(data['A'][0].shape)
    # print(data['B'])
    # print(data['path_A'])  # torch.Size([4, 8, 3, 128, 256])
    # print(data['path_A'][0])  # torch.Size([8, 3, 128, 256])
    data['A'] = data['A'].to(device)
    print('A: ' + str(data['A'].shape))
    print('===========================================================')

    CG = Conv3dGenerator().to(device)
    fake = CG(data['A'])
    print(fake.shape)

    CD = Conv3dDiscriminator().to(device)
    real_or_not = CD(data['A'])
    print(real_or_not.shape)

    FD = FrameDiscriminator_A(batch_size, window_size, step_size).to(device)
    real_or_not_2 = FD(data['A'])
    print(real_or_not_2.shape)

    print("===============generator above")
#
#     fake_judge = LD(fake)
#     print(fake_judge.shape)
#     print(fake_judge.grad_fn)
#
#     fake_indi_judge = iLD(fake)
#     print(fake_indi_judge.shape)
#     print(fake_indi_judge.grad_fn)
#
#     #
#     # y = fake_indi_judge.view(1, 4, 14, 30)
#     # print(y.grad_fn)
#
#
#     # output_list = []
#     # for i in range(fake_judge.shape[0]):
#     #     output_list.append(fake_judge[i])  # add torch.Size([4, 128, 128, 256]) * num_layer
#     # out = torch.cat(output_list, dim=0)  # torch.Size([8, 128, 128, 256])
#     #
#     # print(out.shape)
#     #
#     # return_images = []
#     # for image in fake_judge:
#     #     return_images.append(image)
#     #
#     # out_return = torch.cat(return_images, dim=0)
#     #
#     # print(out_return == out)
#
#
    # convlstm = ConvLSTM(input_size=(128, 256), input_dim=3, hidden_dim=[32, 64], kernel_size=(3, 3), num_layers=2,
    #                     batch_first=True, return_all_layers=True, cuda_use=True).to(device)
    #
    # layer_output_list, last_state_list = convlstm(data['A'].to(device))
    #
    # print('layer below==============')
    # print(layer_output_list[1].shape)  # from layer 1 of lstms / torch.Size([2, 2, 64, 128, 256])
    # out_6 = layer_output_list[1].view([int(batch_size * window_size / step_size), 64, 128, 256])
    # print('out_6: ' + str(out_6.shape))
    # print('======')
    # print(layer_output_list[1][0].shape)  # from layer 2 of lstms / torch.Size([2, 64, 128, 256])
    # print(layer_output_list[1].size()[0])  # from layer 2 of lstms / 2
    # print('======')
    # out = []
    # for i in range(layer_output_list[1].size()[0]):
    #     out.append(layer_output_list[1][i])
    #
    # x = torch.cat(out, dim=0)
    #
    # print(x.shape)
    #
    # out_1 = torch.chunk(x, layer_output_list[1].size()[0], dim=0)
    # out_1_list = list(out_1)
    # out_1_list_stack = torch.stack(out_1_list, dim=0)
    # print(out_1_list_stack)
    #
    # print('===========================================================')

#
#     # out_1, out_2 = torch.chunk(x, layer_output_list[1].size()[0], dim=0)
#     #
#     # print(out_1.size())
#     # print(out_2.size())
#     #
#     # out_3 = torch.stack([out_1, out_2], dim=0)
#     #
#     # print(out_3.size())
#
#     # # gradient check
#     # convlstm = ConvLSTM(input_channels=512, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3, step=5,
#     #                     effective_step=[4]).to(device)
#     # loss_fn = torch.nn.MSELoss()
#     #
#     # input = torch.tensor(torch.randn(1, 2, 3, 16, 64)).to(device)
#     # target = torch.tensor(torch.randn(1, 2, 32, 16, 64)).double().to(device)
#     #
#     # output = convlstm(input)
#     # output = output[0][0].double()
#     # res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
#     # print(res)