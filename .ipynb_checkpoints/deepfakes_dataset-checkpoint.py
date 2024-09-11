import pywt
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2 
import numpy as np
import torch.nn.functional as F
from skimage.transform import resize

import uuid
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, VerticalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate

from transforms.albu import IsotropicResize

def custom_normalize(data):
    # 添加小常数，避免溢出和底数为负数
    data = data/np.max(np.abs(data))
    epsilon = 1e-8
    exp_data = np.exp(-data)
    return (1 - exp_data + epsilon) / (1 + exp_data + epsilon)


def normalized(data):
    target_mean = [0.5,0.5,0.5]
    target_std = [0.5,0.5,0.5]
    # 计算图像当前的均值和标准差
    current_mean = np.mean(data, axis=(0, 1))
    current_std = np.std(data, axis=(0, 1))

    # 标准化图像
    normalized_image = (data - current_mean) / current_std

    # 缩放使得均值和标准差符合目标值
    scaled_image = normalized_image * target_std + target_mean
    return scaled_image

def rgb2ycbcr(rgb_image):
    """convert rgb into ycbcr"""
    if len(rgb_image.shape)!=3 or rgb_image.shape[2]!=3:
        raise ValueError("input image is not a rgb image")
    rgb_image = rgb_image.astype(np.float32)
    # 1：创建变换矩阵，和偏移量
    transform_matrix = np.array([[0.257, 0.504, 0.098],
                                 [-0.148, -0.291, 0.439],
                                 [0.439, -0.368, -0.071]])
    shift_matrix = np.array([16, 128, 128])
    ycbcr_image = np.zeros(shape=rgb_image.shape)
    w, h, _ = rgb_image.shape
    # 2：遍历每个像素点的三个通道进行变换
    for i in range(w):
        for j in range(h):
            ycbcr_image[i, j, :] = np.dot(transform_matrix, rgb_image[i, j, :]) + shift_matrix
    return ycbcr_image

def scharr(yc):
    scharrx = cv2.Scharr(yc, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(yc, cv2.CV_64F, 0, 1)
    scharrx = cv2.convertScaleAbs(scharrx)  # 转回uint8
    scharry = cv2.convertScaleAbs(scharry)
    scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
    return scharrxy

def sigmoid_transform(LL):
    # 对LL进行Sigmoid变换
    LL = LL / np.max(np.abs(LL))
    sigmoid_LL = 1 / (1 + np.exp(-LL))
    return sigmoid_LL

def normalize(x):
    x = np.abs(x)
    x = ((2 * (x - np.max(x))) / (np.max(x) - np.min(x))) + 1
    return x

class DeepFakesDataset(Dataset):
    def __init__(self, images, labels, image_size, mode = 'train'):
        self.x = images
        self.y = torch.from_numpy(labels)
        self.image_size = image_size
        self.mode = mode
        self.n_samples = len(images)

    

        
    def create_train_transforms(self, size):
        return Compose([
            GaussNoise(p=0.1),
            HorizontalFlip(always_apply=False, p=0.1),
            VerticalFlip(always_apply=False, p=0.1),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            # OneOf([
            #     IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            #     IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            #     IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            # ], p=1),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        ]
        )
    def create_val_transform(self, size):
        return Compose([
            GaussNoise(p=0.1),
            HorizontalFlip(always_apply=False, p=0.1),
            VerticalFlip(always_apply=False, p=0.1),
        ]
        )
 
    def create_base_transform(self, size):
        return Compose([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        ])

    


    def __getitem__(self, index):
        global image1, image2
        images = self.x[index]
            
        transform = self.create_train_transforms(self.image_size)
        transform_ = self.create_base_transform(self.image_size)
        
            
        image_1 = []
        image_2 = []
        image_di = 0
        image_temp = 0
        if self.mode != 'test':
            images = transform(image=images)['image']
        else:
            images = transform_(image=images)['image']
        #images = transform(image=images)['image']
        imagel = images
        image_ = cv2.cvtColor(imagel, cv2.COLOR_BGR2GRAY).astype(np.float32)

        ycbcr_image = cv2.cvtColor(imagel, cv2.COLOR_RGB2YCR_CB).astype(np.float32)
        ycb = ycbcr_image[:,:,2]
        ycr = ycbcr_image[:,:,1]

        ycbcr_image[:,:,1] = scharr(ycb)
        ycbcr_image[:,:,2] = scharr(ycr)

        ycbcr_image = cv2.resize(ycbcr_image,(224, 224), interpolation=cv2.INTER_AREA)
        image_ycrcb = custom_normalize(ycbcr_image)


        
        coeffs = pywt.dwt2(image_, 'haar')
        LL, (LH, HL, HH) = coeffs
        LL = np.abs(LL)
        LH_ = cv2.resize(np.abs(LH) + LL/255, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
        HL_ = cv2.resize(np.abs(HL) + LL/255, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
        HH_ = cv2.resize(np.exp(custom_normalize(np.abs(HH))), (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
        
        # LH_ = cv2.resize(np.exp(custom_normalize(np.abs(LH))), (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
        # HL_ = cv2.resize(np.exp(custom_normalize(np.abs(HL))), (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
        # HH_ = cv2.resize(np.exp(custom_normalize(np.abs(HH))), (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
        
        # LH_ = np.abs(LH) + LL/255
        # HL_ = np.abs(HL) + LL/255
        # HH_ = np.exp(np.abs(HH))
        combined_img = np.zeros_like(image_ycrcb, dtype=np.float32)
        combined_img[:, :, 0] = LH_
        combined_img[:, :, 1] = HL_
        combined_img[:, :, 2] = HH_
        
        # combined_img[:, :, 0] = HH_
        # combined_img[:, :, 1] = HH_
        # combined_img[:, :, 2] = HH_
      
        combined_img = cv2.resize(combined_img,(224, 224), interpolation=cv2.INTER_AREA)
        image_haar = custom_normalize(combined_img)

        # image_1.append(image_ycrcb)
        # image_2.append(image_haar)
        # image1 = normalized(image_ycrcb)
        # image2 = normalized(image_haar)
        image1 = image_ycrcb
        image2 = image_haar
        return torch.tensor(image1).float(), torch.tensor(image2).float(), self.y[index]
    #return torch.tensor(image1).float(), torch.tensor(image2).float(), self.y[index]

    def __len__(self):
        return self.n_samples

 
