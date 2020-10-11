import os, glob
import random
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image


class UnalignedDataset(torch.utils.data.Dataset):
    def __init__(self, name_dataset, load_size, crop_size, is_train):
        super(UnalignedDataset, self).__init__()
        self.load_size = load_size
        self.crop_size = crop_size
        self.is_train = is_train

        root_dir = os.path.join('dataset', name_dataset)

        if is_train:
            dir_A = os.path.join(root_dir, 'trainA')
            dir_B = os.path.join(root_dir, 'trainB')
            dir_A_mask = os.path.join(root_dir, 'trainA_mask')

            self.image_paths_A = self._make_dataset(dir_A)
            self.image_paths_B = self._make_dataset(dir_B)
            self.image_paths_A_mask = self._make_dataset(dir_A_mask)

            self.size_A = len(self.image_paths_A)
            self.size_B = len(self.image_paths_B)
            self.size_A_mask = len(self.image_paths_A_mask)

        else:
            dir_A = os.path.join(root_dir, 'testA')
            dir_B = os.path.join(root_dir, 'testB')

            self.image_paths_A = self._make_dataset(dir_A)
            self.image_paths_B = self._make_dataset(dir_B)

            self.size_A = len(self.image_paths_A)
            self.size_B = len(self.image_paths_B)

        self.transform = self._make_transform(load_size, crop_size, is_train)

    # get tensor data
    def __getitem__(self, index):
        index_A = index % self.size_A  # due to the different num of each data A, B
        path_A = self.image_paths_A[index_A]
        img_A = Image.open(path_A).convert('RGB')
        A = self.transform(img_A)


        # sample index B at random
        index_B = random.randint(0, self.size_B - 1)
        path_B = self.image_paths_B[index_B]
        img_B = Image.open(path_B).convert('RGB')
        B = self.transform(img_B)

        if self.is_train:
            index_A_mask = index % self.size_A_mask
            path_A_mask = self.image_paths_A_mask[index_A_mask]
            img_A_mask = Image.open(path_A_mask).convert('RGB')
            A_mask = self.transform(img_A_mask)
            
            return {'A': A, 'B': B, 'A_mask': A_mask, 'path_A': path_A, 'path_B': path_B, 'path_A_mask': path_A_mask}

        return {'A': A, 'B': B, 'path_A': path_A, 'path_B': path_B}

    def __len__(self):
        len = max(self.size_A, self.size_B)
        return len

    def _make_dataset(self, dir):
        images = []
        for fname in os.listdir(dir):
            if fname.endswith('.jpg'):
                path = os.path.join(dir, fname)
                images.append(path)
        return images

    def _make_transform(self, load_size, crop_size, is_train):
        transforms_list = []
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        return transforms.Compose(transforms_list)


if __name__ == '__main__':

    batch_size = 4

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    print('device {}'.format(device))


