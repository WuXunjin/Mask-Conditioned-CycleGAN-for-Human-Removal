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

from dataset import UnalignedDataset
from model_base import Generator, Discriminator


class CycleGAN(object):

    def __init__(self, log_dir='logs', device='cuda:0', lr=0.0002, beta1=0.5,
                 lambda_idt=5.0, lambda_A=10.0, lambda_B=10.0, lambda_mask=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.device = device
        self.gpu_ids = [0, 1]  # 0, 1, 2

        self.netG_A = Generator().to(self.device)
        self.netG_B = Generator().to(self.device)
        self.netD_A = Discriminator().to(self.device)
        self.netD_B = Discriminator().to(self.device)

        print(torch.cuda.is_available())

        # multi-GPUs
        self.netG_A = torch.nn.DataParallel(self.netG_A, self.gpu_ids)
        self.netG_B = torch.nn.DataParallel(self.netG_B, self.gpu_ids)
        self.netD_A = torch.nn.DataParallel(self.netD_A, self.gpu_ids)
        self.netD_B = torch.nn.DataParallel(self.netD_B, self.gpu_ids)

        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)

        # targetが本物か偽物かで代わるのでオリジナルのGANLossクラスを作成
        self.criterionGAN = GANLoss(self.device)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionMask = MASKLoss(self.device)

        # weights of loss function
        self.lambda_idt = lambda_idt
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_mask = lambda_mask

        # Generatorは2つのパラメータを同時に更新
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
            lr=self.lr,
            betas=(self.beta1, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D_A)
        self.optimizers.append(self.optimizer_D_B)

        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def set_input(self, input):
        # self.real_A = input['A']
        # self.real_B = input['B']
        # self.real_A_mask = input['A_mask']

        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_A_mask = input['A_mask'].to(self.device)

        # self.image_paths = input['path_A']

    def backward_G(self, real_A, real_B, real_A_mask):
        # Generatorに関連するlossと勾配計算処理
        # G_A, G_Bは変換先ドメインの本物画像を入力したときはそのまま出力するべき
        # netG_AはドメインAの画像からドメインBの画像を生成するGeneratorだが
        # ドメインBの画像も入れることができる
        # その場合は何も変換してほしくないという制約
        # TODO: idt_Aの命名はよくない気がする idt_Bの方が適切では？
        idt_A = self.netG_A(real_B)
        loss_idt_A = self.criterionIdt(idt_A, real_B) * self.lambda_idt

        idt_B = self.netG_B(real_A)
        loss_idt_B = self.criterionIdt(idt_B, real_A) * self.lambda_idt

        # GAN loss D_A(G_A(A))
        # G_Aとしては生成した偽物画像が本物（True）とみなしてほしい
        fake_B = self.netG_A(real_A)
        pred_fake_B = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake_B, True)

        # GAN loss D_B(G_B(B))
        # G_Bとしては生成した偽物画像が本物（True）とみなしてほしい
        fake_A = self.netG_B(real_B)
        pred_fake_A = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake_A, True)

        # forward cycle loss
        # real_A => fake_B => rec_Aが元のreal_Aに近いほどよい
        rec_A = self.netG_B(fake_B)
        loss_cycle_A = self.criterionCycle(rec_A, real_A) * self.lambda_A

        # backward cycle loss
        # real_B => fake_A => rec_Bが元のreal_Bに近いほどよい
        rec_B = self.netG_A(fake_A)
        loss_cycle_B = self.criterionCycle(rec_B, real_B) * self.lambda_B

        ############################################################################################

        # mse for mase as a new loss function
        if self.lambda_mask == 0:
            loss_mask = torch.tensor(0).to(self.device)
        else:
            loss_mask = self.criterionMask(real_A, fake_B, real_A_mask) * self.lambda_mask

        ############################################################################################

        # combined loss
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B + loss_mask
        loss_G.backward()

        # 次のDiscriminatorの更新でfake画像が必要なので一緒に返す
        return loss_G_A.data, loss_G_B.data, loss_cycle_A.data, loss_cycle_B.data, \
               loss_idt_A.data, loss_idt_B.data, loss_mask.data, fake_A.data, fake_B.data

    def backward_D_A(self, real_B, fake_B):
        # ドメインAから生成したfake_Bが本物か偽物か見分ける

        # TODO: これは何をしている？
        # fake_Bを直接使わずに過去に生成した偽画像から新しくランダムサンプリングしている？
        fake_B = self.fake_B_pool.query(fake_B)

        # 本物画像を入れたときは本物と認識するほうがよい
        pred_real = self.netD_A(real_B)
        loss_D_real = self.criterionGAN(pred_real, True)

        # ドメインAから生成した偽物画像を入れたときは偽物と認識するほうがよい
        # fake_Bを生成したGeneratorまで勾配が伝搬しないようにdetach()する
        pred_fake = self.netD_A(fake_B.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # combined loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        return loss_D_A.data

    def backward_D_B(self, real_A, fake_A):
        # ドメインBから生成したfake_Aが本物か偽物か見分ける

        fake_A = self.fake_A_pool.query(fake_A)

        # 本物画像を入れたときは本物と認識するほうがよい
        pred_real = self.netD_B(real_A)
        loss_D_real = self.criterionGAN(pred_real, True)

        # 偽物画像を入れたときは偽物と認識するほうがよい
        pred_fake = self.netD_B(fake_A.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # combined loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        return loss_D_B.data

    def optimize(self):

        # update Generator (G_A and G_B)
        self.optimizer_G.zero_grad()
        loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, loss_idt_A, loss_idt_B, loss_mask, fake_A, fake_B \
            = self.backward_G(self.real_A, self.real_B, self.real_A_mask)
        self.optimizer_G.step()

        # update D_A
        self.optimizer_D_A.zero_grad()
        loss_D_A = self.backward_D_A(self.real_B, fake_B)
        self.optimizer_D_A.step()

        # update D_B
        self.optimizer_D_B.zero_grad()
        loss_D_B = self.backward_D_B(self.real_A, fake_A)
        self.optimizer_D_B.step()

        ret_loss = [loss_G_A, loss_D_A,
                    loss_G_B, loss_D_B,
                    loss_cycle_A, loss_cycle_B,
                    loss_idt_A, loss_idt_B,
                    loss_mask]

        return np.array(ret_loss)

    def train(self, data_loader):
        running_loss = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        time_list = []
        for batch_idx, data in enumerate(data_loader):

            t1 = time.perf_counter()
            self.set_input(data)
            losses = self.optimize()
            losses = losses.astype(np.float32)
            running_loss += losses

            t2 = time.perf_counter()
            get_processing_time = t2 - t1
            time_list.append(get_processing_time)

            if batch_idx % 500 == 0:
                print('batch: {} / elapsed_time: {} sec'.format(batch_idx, sum(time_list)))
                time_list = []

        running_loss /= len(data_loader)
        return running_loss

    def save_network(self, network, network_label, epoch_label):
        save_filename = '{}_net_{}.pth'.format(epoch_label, network_label)
        save_path = os.path.join(self.log_dir, save_filename)

        #         torch.save({'epoch': epoch_label,
        #                     'model_state_dict': network.cpu().state_dict(),
        #                     'optimizer_state_dict': optimizer.cpu().state_dict(),
        #                     'loss': loss}, save_path)
        #         # GPUに戻す
        #         network.to(device)

        # GPUで動いている場合はCPUに移してから保存
        # これやっておけばCPUでモデルをロードしやすくなる？
        torch.save(network.cpu().state_dict(), save_path)
        # GPUに戻す
        network.to(self.device)

    def load_network(self, network, network_label, epoch_label):
        load_filename = '{}_net_{}.pth'.format(epoch_label, network_label)
        load_path = os.path.join(self.log_dir, load_filename)
        network.load_state_dict(torch.load(load_path))

    #         network = torch.nn.DataParallel(network, self.gpu_ids)

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label)
        self.save_network(self.netD_A, 'D_A', label)
        self.save_network(self.netG_B, 'G_B', label)
        self.save_network(self.netD_B, 'D_B', label)

    def load(self, label):
        self.load_network(self.netG_A, 'G_A', label)
        self.load_network(self.netD_A, 'D_A', label)
        self.load_network(self.netG_B, 'G_B', label)
        self.load_network(self.netD_B, 'D_B', label)


class ImagePool(object):
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        # プールを使わないときはそのまま返す
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            # バッチの次元を削除して3Dテンソルに
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


class GANLoss(nn.Module):

    def __init__(self, device):
        super(GANLoss, self).__init__()
        self.device = device
        self.real_label_var = None
        self.fake_label_var = None
        self.loss = nn.MSELoss()
#         self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
#         self.cuda = torch.cuda.is_available()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            # 高速化のため？
            # varがNoneのままか形状が違うときに作り直す
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = torch.ones(input.size()).to(self.device)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = torch.zeros(input.size()).to(self.device)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class MASKLoss(nn.Module):
    def __init__(self, device):
        super(MASKLoss, self).__init__()
        self.device = device
        self.loss = nn.MSELoss()

    def get_img_with_mask(self, real, fake, real_mask):
        input_1 = torch.tensor([0.1, 0.1, 0.1], requires_grad=False).to(self.device)

        # [-1,1] => [0, 1]
        real_A = 0.5 * (real + 1)
        fake_B = 0.5 * (fake + 1)
        real_A_mask = 0.5 * (real_mask + 1)

        # transpose axis
        real_A = real_A.permute(0, 2, 3, 1)
        fake_B = fake_B.permute(0, 2, 3, 1)
        real_A_mask = real_A_mask.permute(0, 2, 3, 1)

        target_with_mask = torch.where(real_A_mask[:, :, :] > input_1, real_A_mask * 0, real_A).to(self.device)
        fake_with_mask = torch.where(real_A_mask[:, :, :] > input_1, real_A_mask * 0, fake_B).to(self.device)

        return fake_with_mask, target_with_mask

    def __call__(self, real, fake, real_mask):
        fake_with_mask, target_with_mask = self.get_img_with_mask(real, fake, real_mask)
        return self.loss(fake_with_mask, target_with_mask)
