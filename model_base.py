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
from tensorboardX import SummaryWriter
import time
import cv2

from dataset import UnalignedDataset


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
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),

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


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim,
                 hidden_dim, kernel_size, bias=True, mode_train=True):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_size (int, int):
            height x width of input tensor
        input_dim (int):
            number of channels of input tensor
        hidden_dim (int):
            number of channels of hidden state
        kernel_size (int):
            size of filter kernel
        bias (bool):
            add bias or not
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        """
        LSTM: gate_size = 4 * hidden_size
        GRU: gate_size = 3 * hidden_size
        """
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)

        self.mode_train = mode_train

    def forward(self, input_tensor, cur_state):
        """
        input_tensor (Tensor):
            input x
        cur_state (Tensor, Tensor):
            h_out, c_out: the output of the previous state
        return：h_next, c_next
        """
        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # ConvLSTM: concatenate to speed up
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, device, cuda_use):
        if cuda_use:
            return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to('cuda:0'),
                    torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to('cuda:0'))
        else:
            return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to('cuda:0'),
                    torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to('cuda:0'))


class ConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size,
                 num_layers, batch_first=False, bias=True,
                 return_all_layers=False, cuda_use=False, device="cuda:0", mode_train=True):
        """
        Initialize ConvLSTM network.
        Parameters
        ----------
        input_size (int, int):
            height and width of input tensor
        input_dim (int):
            number of channels of input tensor
        hidden_dim (int):
            number of channels of hidden state
        kernel_size (int):
            size of filter kernel
        num_layers (int):
            number of ConvLSTMCells
        batch_first (bool):
            batch first or not
        bias (bool):
            add bias or not
        return_all_layers (bool):
            return all the layers or not
        cuda_use (bool):
            use GPU(s) or not
        """

        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # make sure kernel_size and hidden_dim are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if len(kernel_size) != num_layers or len(hidden_dim) != num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.cuda_use = cuda_use
        self.device = device

        cell_list = []

        # add LSTM Unit to cell_list
        for i in range(0, self.num_layers):
            if i == 0:
                cur_input_dim = self.input_dim
            else:
                cur_input_dim = hidden_dim[i - 1]

            if self.cuda_use:
                cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                              input_dim=cur_input_dim,
                                              hidden_dim=self.hidden_dim[i],
                                              kernel_size=self.kernel_size[i],
                                              bias=self.bias, mode_train=mode_train))
            else:
                cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                              input_dim=cur_input_dim,
                                              hidden_dim=self.hidden_dim[i],
                                              kernel_size=self.kernel_size[i],
                                              bias=self.bias))
            self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape
                (t, b, c, h, w) or
                (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        return: last_state_list, layer_output
        """

        # (t, b, c, h, w) -> (b, t, c, h, w)
        if not self.batch_first:
            input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0), cuda_use=self.cuda_use)

            layer_output_list = []
            last_state_list = []

            seq_len = input_tensor.size(1)
            cur_layer_input = input_tensor

            for layer_idx in range(self.num_layers):

                h, c = hidden_state[layer_idx]
                output_inner = []
                for t in range(seq_len):
                    h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                     cur_state=[h, c])
                    output_inner.append(h)

                layer_output = torch.stack(output_inner, dim=1)
                cur_layer_input = layer_output

                layer_output_list.append(layer_output)
                last_state_list.append([h, c])

            if not self.return_all_layers:
                layer_output_list = layer_output_list[-1:]
                last_state_list = last_state_list[-1:]

            return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, cuda_use=False):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, self.device, cuda_use=cuda_use))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class Reshape(nn.Module):
    def __init__(self, args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class LSTMGenerator(nn.Module):
    def __init__(self, batch_size, window_size, step_size, device):
        super(LSTMGenerator, self).__init__()

        self.model_1 = Reshape([int(batch_size * window_size / step_size), 3, 128, 256])

        self.model_2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )

        self.model_3 = Reshape([int(batch_size), int(window_size / step_size), 128, 32, 64])

        self.model_4 = nn.Sequential(
            ConvLSTM(input_size=(32, 64), input_dim=128, hidden_dim=[256], kernel_size=(3, 3), num_layers=1,
                     batch_first=True, return_all_layers=True, cuda_use=True, device=device)
        )

        self.model_5 = Reshape([int(batch_size * window_size / step_size), 256, 32, 64])

        self.model_6 = nn.Sequential(
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256)
        )

        self.model_7 = Reshape([int(batch_size), int(window_size / step_size), 256, 32, 64])

        self.model_8 = nn.Sequential(
            ConvLSTM(input_size=(32, 64), input_dim=256, hidden_dim=[128], kernel_size=(3, 3), num_layers=1,
                     batch_first=True, return_all_layers=True, cuda_use=True, device=device)
        )

        self.model_9 = Reshape([int(batch_size * window_size / step_size), 128, 32, 64])

        self.model_10 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )

        self.model_11 = Reshape([int(batch_size), int(window_size / step_size), 3, 128, 256])

        # initialize weights
        self.model_2.apply(self._init_weights)
        self.model_6.apply(self._init_weights)
        self.model_10.apply(self._init_weights)

    def forward(self, input_img):
        out = self.model_1(input_img)
        out = self.model_2(out)
        out = self.model_3(out)
        layer_output_list_1, _ = self.model_4(out)
        out = self.model_5(layer_output_list_1[0])
        out = self.model_6(out)
        out = self.model_7(out)
        layer_output_list_2, _ = self.model_8(out)
        out = self.model_9(layer_output_list_2[0])
        out = self.model_10(out)
        out = self.model_11(out)
        return out

    @staticmethod
    def _init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal(m.weight.data, 0.0, 0.02)

    @staticmethod
    def mix_batch_and_sequence(layer_output_list):
        output_list = []
        for s in range(layer_output_list[1].size()[0]):
            output_list.append(layer_output_list[1][s])  # add torch.Size([4, 128, 128, 256]) * num_layer
        out = torch.cat(output_list, dim=0)  # torch.Size([8, 128, 128, 256])
        return out
    
    @staticmethod
    def make_batch_based_sequence(sequence, layer_output_list):
        out_taple = torch.chunk(sequence, layer_output_list[1].size()[0], dim=0)
        out_list = list(out_taple)
        out = torch.stack(out_list, dim=0)
        return out


class LSTMDiscriminator(nn.Module):

    def __init__(self, batch_size, window_size, step_size, device):
        super(LSTMDiscriminator, self).__init__()

        self.model_1 = nn.Sequential(
            ConvLSTM(input_size=(128, 256), input_dim=3, hidden_dim=[64, 64], kernel_size=(3, 3), num_layers=2,
                       batch_first=True, return_all_layers=True, cuda_use=True, device=device)
        )

        self.model_2 = nn.Sequential(
            nn.Linear(int(window_size / step_size)*128*256*64, 1),
            nn.Sigmoid()
        )

    # initialize weights
    @staticmethod
    def _init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal(m.weight.data, 0.0, 0.02)

    def forward(self, input_img):
        layer_output_list, _ = self.model_1(input_img)
        out = self.model_2(layer_output_list[1]).squeeze()
        return out


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


