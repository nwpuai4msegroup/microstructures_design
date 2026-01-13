# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 14:07:41 2022

@author: l1415
"""
import torchvision
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch import nn
import cv2
import matplotlib.pyplot as plt
import random
import torch

# 从大图片中随机选择小图片
def random_cut_image(image, img_size, num_cut=20):      #宽为img_size的正方形图片
    width, height = image.size
    idx_width = np.random.randint(0, width-img_size, num_cut)
    idx_height = np.random.randint(0, height-img_size, num_cut)
    box_list = []
    # (left, upper, right, lower)
    for i in range(num_cut):
        box = (idx_width[i], idx_height[i], idx_width[i]+img_size, idx_height[i]+img_size)
        box_list.append(box)
    image_list = [image.crop(box) for box in box_list]
    return image_list

#图片保存
def save_images(image_list,dir_name,file_name):
    index = 1
    for image in image_list:
        image.save(dir_name + '/' + file_name+'_' + str(index) + '.jpg')
        index += 1

class Mydata (Dataset):
    def __init__(self, root_dir, label):
        self.root_dir = root_dir
        #self.label = label
        self.img_path = os.listdir(self.root_dir)
        self.label = label

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_item_path)
        #img = img.resize((178, 218))
        #tensor_trans = torchvision.transforms.ToTensor()
        tensor_trans = torchvision.transforms.Compose([torchvision.transforms.Resize(128), 
                                torchvision.transforms.ToTensor()])
        img = tensor_trans(img)
        i = idx//20
        label = self.label[i]
        return img, label
    
    def __len__(self):
        return len(self.img_path)
class image_data(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        #self.label = label
        self.img_path = os.listdir(self.root_dir)
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_item_path)
        #img = img.resize((178, 218))
        #tensor_trans = torchvision.transforms.ToTensor()
        tensor_trans = torchvision.transforms.Compose([torchvision.transforms.Resize(128), torchvision.transforms.ToTensor()])
        img = tensor_trans(img)
        return img
    def __len__(self):
        return len(self.img_path)

class image_data_min(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        #self.label = label
        self.img_path = os.listdir(self.root_dir)
        self.img_path_min = []
        for i in range(len(self.img_path)):
            img_name = self.img_path[i]
            if img_name[:2] in ["05", "06", "07", "08", "09", "10", "19", "29"]:
                self.img_path_min.append(img_name)
    def __getitem__(self, idx):
        img_idx = self.img_path_min[idx]
        img_item_path = os.path.join(self.root_dir, img_idx)
        img = Image.open(img_item_path)
        tensor_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
            torchvision.transforms.Resize(128),
            #torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        img = tensor_trans(img)
        return img
    def __len__(self):
        return len(self.img_path_min)

class image_data_max(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        #self.label = label
        self.img_path = os.listdir(self.root_dir)
        self.img_path_max = []
        for i in range(len(self.img_path)):
            img_name = self.img_path[i]
            if img_name[:2] in ["15", "24", "25", "26"]:
                self.img_path_max.append(img_name)
    def __getitem__(self, idx):
        img_idx = self.img_path_max[idx]
        img_item_path = os.path.join(self.root_dir, img_idx)
        img = Image.open(img_item_path)
        tensor_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
            torchvision.transforms.Resize(128),
            #torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        img = tensor_trans(img)
        return img
    def __len__(self):
        return len(self.img_path_max)
#图像增强，还需增加其它方法
def imgextend(img):
    ansimg = torch.empty(img.shape)
    for i in range(img.shape[0]):
        imgi = img[i]
        flag = random.random() #随机翻转
        imgi = imgi.permute((1, 2, 0)).numpy()# transpose只能交换两个维度
        width, height, _ = imgi.shape
        if 0 <= flag < 0.1:
            imgi = cv2.flip(imgi, 1) #水平翻转函数
        elif 0.1 <= flag < 0.2:#上下翻转
            imgi = cv2.flip(imgi, 0)
        elif 0.2 <= flag < 0.3:
            rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
            imgi = cv2.warpAffine(imgi, rotationMatrix, (width, height))
        elif 0.3 <= flag < 0.4:
            rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), 180, 1)
            imgi = cv2.warpAffine(imgi, rotationMatrix, (width, height))
        elif 0.4 <= flag < 0.5:
            rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), 270, 1)
            imgi = cv2.warpAffine(imgi, rotationMatrix, (width, height))
        elif 0.5 <= flag < 0.6:
            idx_width = np.random.randint(0, width-20, 3)
            idx_height = np.random.randint(0, height-20, 3)
            imgi_1 = imgi.copy()
            imgi_1[idx_width[0]:idx_width[0]+20, idx_height[0]:idx_height[0]+20] = 0
            imgi_1[idx_width[1]:idx_width[1]+20, idx_height[1]:idx_height[1]+20] = 0
            imgi_1[idx_width[2]:idx_width[2]+20, idx_height[2]:idx_height[2]+20] = 0
            imgi = imgi_1
        elif 0.6 <= flag < 0.7:
            imgi = cv2.GaussianBlur(imgi, (5, 5), 5)
        elif 0.7 <= flag < 0.8:
            pts1 = np.float32([[50,50],[200,50],[50,200]])
            pts2 = np.float32([[55,55],[190,50],[70,220]])
            M = cv2.getAffineTransform(pts1,pts2)
            imgi = cv2.warpAffine(imgi,M,(width,height))
        elif 0.8 <= flag < 0.9:
            kenel = np.ones((2, 2), np.uint8)
            imgi = cv2.dilate(imgi, kenel)
        elif 0.9 <= flag < 1:
            kenel = np.ones((2, 2), np.uint8)
            imgi = cv2.erode(imgi, kenel)
        else:
            imgi = imgi
        toten = torchvision.transforms.Compose([torchvision.transforms.Resize(128), torchvision.transforms.ToTensor()])
        imgi = toten(imgi)
        ansimg[i]= imgi
    return ansimg


#反转和旋转变化
def imgextend_1(img):
    ansimg = torch.empty(img.shape)
    for i in range(img.shape[0]):
        imgi = img[i]
        flag = random.random()*1.2 #随机
        imgi = imgi.permute((1, 2, 0)).numpy()# transpose只能交换两个维度
        width, height, _ = imgi.shape
        if 0 <= flag < 0.2:
            imgi = cv2.flip(imgi, 1) #水平翻转函数
        elif 0.2 <= flag < 0.4:#上下翻转
            imgi = cv2.flip(imgi, 0)
        elif 0.4 <= flag < 0.6:#二维旋转
            rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
            imgi = cv2.warpAffine(imgi, rotationMatrix, (width, height))
        elif 0.6 <= flag < 0.8:#二维旋转
            rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), 180, 1)
            imgi = cv2.warpAffine(imgi, rotationMatrix, (width, height))
        elif 0.8 <= flag < 1.0:#二维旋转
            rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), 270, 1)
            imgi = cv2.warpAffine(imgi, rotationMatrix, (width, height))
        else:
            imgi = imgi
        toten = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(128)])
        imgi = toten(imgi)
        ansimg[i]= imgi
    return ansimg

def imgextend_2(img):
    ansimg = torch.empty(img.shape)
    for i in range(img.shape[0]):
        imgi = img[i]
        flag = random.random() #随机翻转
        imgi = imgi.permute((1, 2, 0)).numpy()# transpose只能交换两个维度
        width, height, _ = imgi.shape
        if 0 <= flag < 0.1:#随机添加三块遮挡
            idx_width = np.random.randint(0, width-20, 3)
            idx_height = np.random.randint(0, height-20, 3)
            imgi_1 = imgi.copy()
            imgi_1[idx_width[0]:idx_width[0]+20, idx_height[0]:idx_height[0]+20] = 0
            imgi_1[idx_width[1]:idx_width[1]+20, idx_height[1]:idx_height[1]+20] = 0
            imgi_1[idx_width[2]:idx_width[2]+20, idx_height[2]:idx_height[2]+20] = 0
            imgi = imgi_1
        elif 0.1 <= flag < 0.2:#高斯模糊
            imgi = cv2.GaussianBlur(imgi, (5, 5), 5)
        elif 0.2 <= flag < 0.3:#仿射变换
            pts1 = np.float32([[50,50],[200,50],[50,200]])
            pts2 = np.float32([[55,55],[190,50],[70,220]])
            M = cv2.getAffineTransform(pts1,pts2)
            imgi = cv2.warpAffine(imgi,M,(width,height))
        elif 0.3 <= flag < 0.4:#膨胀
            kenel = np.ones((2, 2), np.uint8)
            imgi = cv2.dilate(imgi, kenel)
        elif 0.4 <= flag < 0.5:#腐蚀
            kenel = np.ones((2, 2), np.uint8)
            imgi = cv2.erode(imgi, kenel)
        else:
            imgi = imgi
        toten = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(128)])
        imgi = toten(imgi)
        ansimg[i]= imgi
    return ansimg

#膨胀变化


def imgextend_erode(img):
    ansimg = torch.empty(img.shape)
    for i in range(img.shape[0]):
        imgi = img[i]
        flag = random.random() #随机翻转
        imgi = imgi.permute((1, 2, 0)).numpy()# transpose只能交换两个维度
        width, height, _ = imgi.shape
        kenel = np.ones((2, 2), np.uint8)
        imgi = cv2.erode(imgi, kenel)
        toten = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(128)])
        imgi = toten(imgi)
        ansimg[i]= imgi
    return ansimg

def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, n_channels):
        super(UnFlatten, self).__init__()
        self.n_channels = n_channels

    def forward(self, input):
        size = int((input.size(1) // self.n_channels) ** 0.5)
        return input.view(input.size(0), self.n_channels, size, size)


class SigmaVAE(nn.Module):
    def __init__(self, device='cuda', img_channels=3, args=None):
        super().__init__()
        self.batch_size = args.batch_size
        self.device = device
        self.z_dim = 128
        self.img_channels = img_channels
        self.model = args.model
        img_size = 128
        filters_m = 32

        ## Build network
        self.encoder = self.get_encoder(self.img_channels, filters_m)

        # output size depends on input image size, compute the output size
        demo_input = torch.ones([1, self.img_channels, img_size, img_size])
        h_dim = self.encoder(demo_input).shape[1]
        # print(h_dim)
        # map to latent z
        self.fc11 = nn.Linear(h_dim, self.z_dim)
        self.fc12 = nn.Linear(h_dim, self.z_dim)

        # decoder
        self.fc2 = nn.Linear(self.z_dim, h_dim)
        self.decoder = self.get_decoder(filters_m, self.img_channels)

        self.log_sigma = 0
        if self.model == 'sigma_vae':
            ## Sigma VAE
            self.log_sigma = torch.nn.Parameter(torch.full((1,), 0)[0].float(), requires_grad=args.model == 'sigma_vae')

#     @staticmethod
#     def get_encoder(img_channels, filters_m):
#         return nn.Sequential(
#             nn.Conv2d(img_channels, filters_m, (3, 3), stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(filters_m, 2 * filters_m, (3, 3), stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(2 * filters_m, 2 * filters_m, (4, 4), stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(2 * filters_m, 4 * filters_m, (4, 4), stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(4 * filters_m, 8 * filters_m, (5, 5), stride=2, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(8 * filters_m, 8 * filters_m, (5, 5), stride=2, padding=2),
#             nn.ReLU(),
#             Flatten()
#         )

#     @staticmethod
#     def get_decoder(filters_m, out_channels):
#         return nn.Sequential(
#             UnFlatten(8 * filters_m),
#             nn.ConvTranspose2d(8 * filters_m, 8 * filters_m, (6, 6), stride=2, padding=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(8 * filters_m, 4 * filters_m, (6, 6), stride=2, padding=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(4 * filters_m, 2 * filters_m, (6, 6), stride=2, padding=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(2 * filters_m, 2 * filters_m, (6, 6), stride=2, padding=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(2 * filters_m, filters_m, (5, 5), stride=1, padding=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(filters_m, out_channels, (5, 5), stride=1, padding=2),
#             nn.Sigmoid(),
#         )

    @staticmethod
    def get_encoder(img_channels, filters_m):
        return nn.Sequential(
            nn.Conv2d(img_channels, filters_m, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters_m, 2 * filters_m, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * filters_m, 4 * filters_m, (5, 5), stride=2, padding=2),
            nn.ReLU(),
            Flatten()
        )

    @staticmethod
    def get_decoder(filters_m, out_channels):
        return nn.Sequential(
            UnFlatten(4 * filters_m),
            nn.ConvTranspose2d(4 * filters_m, 2 * filters_m, (6, 6), stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * filters_m, filters_m, (6, 6), stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(filters_m, out_channels, (5, 5), stride=1, padding=2),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        # print(h.shape)
        return self.fc11(h), self.fc12(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # print(self.fc2(z).shape)
        return self.decoder(self.fc2(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        #print(z.shape)
        return self.decode(z), mu, logvar

    def sample(self, n):
        sample = torch.randn(n, self.z_dim).to(self.device)
        return self.decode(sample)

    def reconstruction_loss(self, x_hat, x):
        """ Computes the likelihood of the data given the latent variable,
        in this case using a Gaussian distribution with mean predicted by the neural network and variance = 1 """

        if self.model == 'gaussian_vae':
            # Naive gaussian VAE uses a constant variance
            log_sigma = torch.zeros([], device=x_hat.device)
        elif self.model == 'sigma_vae':
            # Sigma VAE learns the variance of the decoder as another parameter
            log_sigma = self.log_sigma
        elif self.model == 'optimal_sigma_vae':
            log_sigma = ((x - x_hat) ** 2).mean([0,1,2,3], keepdim=True).sqrt().log()
            self.log_sigma = log_sigma.item()
        else:
            raise NotImplementedError

        # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
        # ensures stable training.
        log_sigma = softclip(log_sigma, -6)

        rec = gaussian_nll(x_hat, log_sigma, x).sum()

        return rec

    def loss_function(self, recon_x, x, mu, logvar):
        # Important: both reconstruction and KL divergence loss have to be summed over all element!
        # Here we also sum the over batch and divide by the number of elements in the data later
        if self.model == 'mse_vae':
            rec = torch.nn.MSELoss()(recon_x, x)
        else:
            rec = self.reconstruction_loss(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return rec, kl

class SigmaVAE_new(nn.Module):
    def __init__(self, device='cuda', img_channels=3, args=None):
        super().__init__()
        self.batch_size = args.batch_size
        self.device = device
        self.z_dim = 128
        self.img_channels = img_channels
        self.model = args.model
        img_size = 128
        filters_m = 32

        ## Build network
        self.encoder = self.get_encoder(self.img_channels, filters_m)

        # output size depends on input image size, compute the output size
        demo_input = torch.ones([1, self.img_channels, img_size, img_size])
        h_dim = self.encoder(demo_input).shape[1]
        # print(h_dim)
        # map to latent z
        self.fc11 = nn.Linear(h_dim, self.z_dim)
        self.fc12 = nn.Linear(h_dim, self.z_dim)

        # decoder
        self.fc2 = nn.Linear(self.z_dim, h_dim)
        self.decoder = self.get_decoder(filters_m, self.img_channels)

        self.log_sigma = 0
        if self.model == 'sigma_vae':
            ## Sigma VAE
            self.log_sigma = torch.nn.Parameter(torch.full((1,), 0)[0].float(), requires_grad=args.model == 'sigma_vae')

    @staticmethod
    def get_encoder(img_channels, filters_m):
        return nn.Sequential(
            nn.Conv2d(img_channels, filters_m, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters_m, 2 * filters_m, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * filters_m, 2 * filters_m, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * filters_m, 4 * filters_m, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4 * filters_m, 8 * filters_m, (5, 5), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8 * filters_m, 8 * filters_m, (5, 5), stride=2, padding=2),
            nn.ReLU(),
            Flatten()
        )

    @staticmethod
    def get_decoder(filters_m, out_channels):
        return nn.Sequential(
            UnFlatten(8 * filters_m),
            nn.ConvTranspose2d(8 * filters_m, 8 * filters_m, (6, 6), stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8 * filters_m, 4 * filters_m, (6, 6), stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4 * filters_m, 2 * filters_m, (6, 6), stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * filters_m, 2 * filters_m, (6, 6), stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * filters_m, filters_m, (5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(filters_m, out_channels, (5, 5), stride=1, padding=2),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        # print(h.shape)
        return self.fc11(h), self.fc12(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # print(self.fc2(z).shape)
        return self.decoder(self.fc2(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        #print(z.shape)
        return self.decode(z), mu, logvar

    def sample(self, n):
        sample = torch.randn(n, self.z_dim).to(self.device)
        return self.decode(sample)

    def reconstruction_loss(self, x_hat, x):
        """ Computes the likelihood of the data given the latent variable,
        in this case using a Gaussian distribution with mean predicted by the neural network and variance = 1 """

        if self.model == 'gaussian_vae':
            # Naive gaussian VAE uses a constant variance
            log_sigma = torch.zeros([], device=x_hat.device)
        elif self.model == 'sigma_vae':
            # Sigma VAE learns the variance of the decoder as another parameter
            log_sigma = self.log_sigma
        elif self.model == 'optimal_sigma_vae':
            log_sigma = ((x - x_hat) ** 2).mean([0,1,2,3], keepdim=True).sqrt().log()
            self.log_sigma = log_sigma.item()
        else:
            raise NotImplementedError

        # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
        # ensures stable training.
        log_sigma = softclip(log_sigma, -6)

        rec = gaussian_nll(x_hat, log_sigma, x).sum()

        return rec

    def loss_function(self, recon_x, x, mu, logvar):
        # Important: both reconstruction and KL divergence loss have to be summed over all element!
        # Here we also sum the over batch and divide by the number of elements in the data later
        if self.model == 'mse_vae':
            rec = torch.nn.MSELoss()(recon_x, x)
        else:
            rec = self.reconstruction_loss(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return rec, kl

def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)


class Residual_unet(torch.nn.Module):

    def __init__(self, channel_in, channel_out):
        super().__init__()

        self.cnn = torch.nn.Conv2d(channel_in,
                                   channel_out,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

        self.s = torch.nn.Sequential(
            torch.nn.BatchNorm2d(channel_in),
            torch.nn.Conv2d(channel_in,
                            channel_out,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(channel_out,
                            channel_out,
                            kernel_size=3,
                            stride=1,
                            padding=1),
        )

    def forward(self, x):
        return self.cnn(x) + self.s(x)
   
class UNet_vae(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.down = torch.nn.ModuleList([
            Residual_unet(3, 32),
            Residual_unet(32, 32),
            torch.nn.AvgPool2d(2),
            Residual_unet(32, 64),
            Residual_unet(64, 64),
            torch.nn.AvgPool2d(2),
            Residual_unet(64, 96),
            Residual_unet(96, 96),
            torch.nn.AvgPool2d(2),
        ])

        self.middle = torch.nn.ModuleList([
            Residual_unet(96, 128),
            Residual_unet(128, 128),
        ])

        self.up = torch.nn.ModuleList([
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            Residual_unet(224, 96),
            Residual_unet(192, 96),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            Residual_unet(160, 64),
            Residual_unet(128, 64),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            Residual_unet(96, 32),
            Residual_unet(64, 32),
        ])

        self.out = torch.nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, image):
        #image -> [b, 64, 64, 64]

        #[b, 32, 64, 64]
        #[b, 32, 64, 64]
        #[b, 64, 32, 32]
        #[b, 64, 32, 32]
        #[b, 96, 16, 16]
        #[b, 96, 16, 16]
        out = []
        #image = torch.cat((image, out_vae), dim=1)
        for i in self.down:
            image = i(image)
            if type(i) == Residual_unet:
                out.append(image)

        #[b, 96, 16, 16] -> [b, 128, 8, 8]
        for i in self.middle:
            image = i(image)

        #[b, 96, 16, 16]
        #[b, 96, 16, 16]
        #[b, 64, 32, 32]
        #[b, 64, 32, 32]
        #[b, 32, 64, 64]
        #[b, 32, 64, 64]
        for i in self.up:
            if type(i) == Residual_unet:
                #[b, 128+96, 16, 16] -> [b, 224, 16, 16]
                #[b, 96+96, 16, 16] -> [b, 192, 16, 16]
                #[b, 96+64, 32, 32] -> [b, 160, 32, 32]
                #[b, 64+64, 32, 32] -> [b, 128, 32, 32]
                #[b, 64+32, 64, 64] -> [b, 96, 64, 64]
                #[b, 32+32, 64, 64] -> [b, 64, 64, 64]
                p = out.pop()
                image = torch.cat((image, p), dim=1)
            image = i(image)

        #[b, 32, 64, 64] -> [b, 3, 64, 64]
        image = self.out(image)

        return image


#UNet()(torch.randn(2, 3, 128, 128)).shape
#unet = UNet()

def schedule(time, method='offset_cosine'):
    if method == 'linear':
        t = 1 - (1e-4 + time * (0.02 - 1e-4))

        #累乘
        #cumprod = [1]
        #for i in t:
        #    cumprod.append(cumprod[-1] * i)
        #t = torch.FloatTensor(cumprod[1:])

        #等价
        t = torch.cumprod(t, dim=0)

        return (1 - t).sqrt(), t.sqrt()

    if method == 'cosine':
        #1.5707963267948966 = pi/2
        t = time * 1.5707963267948966
        return t.sin(), t.cos()

    if method == 'offset_cosine':
        #0.3175604292915215 = acos(0.95)
        #1.2332345639299847 = acos(0.02) - acos(0.95)
        t = 0.3175604292915215 + time * 1.2332345639299847

        return t.sin(), t.cos()
class Combine(torch.nn.Module):

    def __init__(self):
        super().__init__()

        #6.907755278982137 = log(1000)
        t = torch.linspace(0.0, 6.907755278982137, 3).exp()
        t *= 2
        #3.141592653589793 = pi
        t *= 3.141592653589793
        self.register_buffer('t', t)

        self.upsample = torch.nn.UpsamplingNearest2d(size=(128, 128))

        self.cnn_img = torch.nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0)
        self.cnn_vae = torch.nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0)

    def get_var(self, var):
        #[b, 1] -> [b, 16]
        var = self.t * var
#         print(self.t.shape, var.shape)

        #[b, 16+16] -> [b, 32]
        var = torch.cat((var.sin(), var.cos()), dim=1)

        #[b, 32] -> [b, 32, 1, 1]
        var = var.unsqueeze(dim=-1).unsqueeze(dim=-1)

        #[b, 32, 1, 1] -> [b, 32, 64, 64]
        var = self.upsample(var)

        return var

    def get_image(self, image, out_vae):
        #[b, 3, 64, 64] -> [b, 32, 64, 64]
        image = self.cnn_img(image)
        out_vae = self.cnn_vae(out_vae)

        return image, out_vae

    def forward(self, image, var, out_vae):
        #image -> [b, 3, 64, 64]
        #var -> [b, 1, 1, 1]

        #[b, 1, 1, 1] -> [b, 1]
        var = var.squeeze(dim=-1).squeeze(dim=-1)

        #[b, 1] -> [b, 32, 64, 64]
        var = self.get_var(var)

        #[b, 3, 64, 64] -> [b, 32, 64, 64]
#         image, out_vae = self.get_image(image, out_vae)

        #[b, 32+32, 64, 64] -> [b, 64, 64, 64]
        combine = torch.cat((image, var, out_vae), dim=1)

        return combine
    
class Residual(torch.nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()

        self.cnn = torch.nn.Conv2d(channel_in,
                                   channel_out,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

        self.s = torch.nn.Sequential(
            torch.nn.BatchNorm2d(channel_in),
            torch.nn.Conv2d(channel_in,
                            channel_out,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(channel_out,
                            channel_out,
                            kernel_size=3,
                            stride=1,
                            padding=1),
        )

    def forward(self, x):
        return self.cnn(x) + self.s(x)
    
    
class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.down = torch.nn.ModuleList([
            Residual(12, 32),
            Residual(32, 32),
            torch.nn.AvgPool2d(2),
            Residual(32, 64),
            Residual(64, 64),
            torch.nn.AvgPool2d(2),
            Residual(64, 96),
            Residual(96, 96),
            torch.nn.AvgPool2d(2),
        ])

        self.middle = torch.nn.ModuleList([
            Residual(96, 128),
            Residual(128, 128),
        ])

        self.up = torch.nn.ModuleList([
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            Residual(224, 96),
            Residual(192, 96),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            Residual(160, 64),
            Residual(128, 64),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            Residual(96, 32),
            Residual(64, 32),
        ])

        self.out = torch.nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, image):#, out_vae
        #image -> [b, 64, 64, 64]

        #[b, 32, 64, 64]
        #[b, 32, 64, 64]
        #[b, 64, 32, 32]
        #[b, 64, 32, 32]
        #[b, 96, 16, 16]
        #[b, 96, 16, 16]
        out = []
#         image = torch.cat((image, out_vae), dim=1)
        for i in self.down:
            image = i(image)
            if type(i) == Residual:
                out.append(image)

        #[b, 96, 16, 16] -> [b, 128, 8, 8]
        for i in self.middle:
            image = i(image)

        #[b, 96, 16, 16]
        #[b, 96, 16, 16]
        #[b, 64, 32, 32]
        #[b, 64, 32, 32]
        #[b, 32, 64, 64]
        #[b, 32, 64, 64]
        for i in self.up:
            if type(i) == Residual:
                #[b, 128+96, 16, 16] -> [b, 224, 16, 16]
                #[b, 96+96, 16, 16] -> [b, 192, 16, 16]
                #[b, 96+64, 32, 32] -> [b, 160, 32, 32]
                #[b, 64+64, 32, 32] -> [b, 128, 32, 32]
                #[b, 64+32, 64, 64] -> [b, 96, 64, 64]
                #[b, 32+32, 64, 64] -> [b, 64, 64, 64]
                p = out.pop()
                image = torch.cat((image, p), dim=1)
            image = i(image)

        #[b, 32, 64, 64] -> [b, 3, 64, 64]
        image = self.out(image)

        return image
from transformers import PreTrainedModel, PretrainedConfig


class Diffusion(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)

        self.norm = torch.nn.BatchNorm2d(3, affine=False)
        self.unet = UNet()
        self.combine = Combine()

    def forward(self, image, out_vae):
        #image -> [b, 3, 64, 64]
        b = image.shape[0]

        #对图像正则化处理
#         image = self.norm(image)
#         out_vae = self.norm(out_vae)

        #随机噪声
        noise = torch.randn(b, 3, 128, 128, device=image.device)
        #noise = noise + out_vae
        
        #随机系数
        #[b, 1, 1, 1],[b, 1, 1, 1]
        noise_r, image_r = schedule(torch.rand(b, 1, 1, 1,
                                               device=image.device))

        #合并图像和噪声
        #[b, 3, 64, 64]
        image = image * image_r + noise * noise_r

        #合并噪声图和噪声系数
        #[b, 64, 64, 64]
        combine = self.combine(image, noise_r**2, out_vae)

        #从噪声图中预测出噪声
        pred_noise = self.unet(combine)

        return noise, pred_noise
def show(images):
    
    if type(images) == torch.Tensor:
        images = images.to('cpu').detach().numpy()

    images = images[:50]

    plt.figure(figsize=(10, 5), dpi=200)

    for i in range(len(images)):
        image = images[i]
        image = image.transpose(1, 2, 0)

        plt.subplot(5, 10, i + 1)
        plt.imshow(image)
        plt.axis('off')

    plt.show()
    