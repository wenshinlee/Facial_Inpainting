import os
import random
import numpy as np
from itertools import chain

import torch
from torch.utils import data
from torchvision import transforms

import cv2
from PIL import Image


class CelebA_Dataset(data.Dataset):
    def __init__(self, image_dir, mask_dir, image_size,
                 facial_fea_names, facial_fea_attr_names, facial_fea_attr_len,
                 facial_attr_dataset, p_irregular_miss, max_num_miss, dilate_iter,
                 is_train, image_transform, mask_transform):
        super(CelebA_Dataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.max_num_miss = max_num_miss
        self.p_irregular_miss = p_irregular_miss
        self.dilate_iter = dilate_iter
        self.is_train = is_train
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.facial_fea_names = facial_fea_names
        self.num_facial_fea = len(self.facial_fea_names)
        self.facial_fea_attr_names = facial_fea_attr_names
        self.facial_fea_attr_len = facial_fea_attr_len
        self.dataset = facial_attr_dataset
        self.dataset = self.dataset[1999:] if self.is_train else self.dataset[:1999]

        assert self.max_num_miss <= self.num_facial_fea, "max_num_miss must <= num_facial_fea!"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        raise NotImplementedError('__getitem__() must be implemented!')

    def load_image(self, file_name):
        image = cv2.imread(os.path.join(self.image_dir, file_name))
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return image

    def load_mask(self, file_name, mask_zeros, miss_area_names=None):
        # Missing based on semantic segmentation map
        if miss_area_names is not None and isinstance(miss_area_names, list):
            # not None, just for specified facial fea miss
            for miss_type in miss_area_names:
                mask_path = os.path.join(self.mask_dir, miss_type, file_name)
                if os.path.exists(mask_path):
                    mask = cv2.resize(cv2.imread(mask_path), (self.image_size, self.image_size),
                                      interpolation=cv2.INTER_NEAREST)

                    # miss area dilate
                    kernel_size = np.random.randint(2, 5)
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=self.dilate_iter)
                    # add to mask_zeros
                    mask_zeros += mask
            mask_zeros = mask_zeros.astype(np.uint8)
        else:  # Random facial fea area miss
            mask_zeros = generate_stroke_mask(mask_zeros)
            mask_zeros = (mask_zeros > 0).astype(np.uint8) * 255

        return mask_zeros

    def load_segmap(self, file_name):
        segmap = np.zeros((self.num_facial_fea, self.image_size, self.image_size))
        for idx, facial_fea_name in enumerate(self.facial_fea_names):
            segmap_path = os.path.join(self.mask_dir, facial_fea_name, file_name)
            if os.path.exists(segmap_path):
                facial_segmap = cv2.resize(cv2.imread(segmap_path, cv2.IMREAD_GRAYSCALE),
                                           (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                segmap[idx, :, :] = facial_segmap / 255.0
        return segmap

    def get_miss_area_names(self, p=None):
        # Equal probability missing
        if not p:
            p_factor = 1.0 / self.num_facial_fea
            p = [p_factor for _ in range(self.num_facial_fea)]
        else:
            assert len(p) == self.num_facial_fea, "p length must equal num_facial_fea"

        np.random.seed()
        num_miss_area = np.random.randint(1, self.max_num_miss + 1)
        miss_area_names = np.random.choice(self.facial_fea_names, num_miss_area, replace=False, p=p).tolist()

        if self.p_irregular_miss > np.random.random(1):  # [0.0, 1.0)
            return None
        else:
            return miss_area_names

    @staticmethod
    def convert_pil(np_image):
        pil_image = Image.fromarray(cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB))
        return pil_image


class CelebA_Attr_CV2(CelebA_Dataset):
    """
    根据语义分割图和随机生成Mask,随机的对图像进行缺失,不考虑属性的问题
    """

    def __init__(self, image_dir, mask_dir, image_size, facial_fea_names, facial_fea_attr_names, facial_fea_attr_len,
                 facial_attr_dataset, p_irregular_miss, max_num_miss, dilate_iter, is_train, image_transform,
                 mask_transform):
        super(CelebA_Attr_CV2, self).__init__(image_dir, mask_dir, image_size, facial_fea_names, facial_fea_attr_names,
                                              facial_fea_attr_len, facial_attr_dataset, p_irregular_miss, max_num_miss,
                                              dilate_iter, is_train, image_transform, mask_transform)

    def __getitem__(self, item):
        file_name, attr_matrix = self.dataset[item]
        image = self.load_image(file_name=file_name)

        mask_zeros = np.zeros_like(image)
        mask = self.load_mask(file_name=file_name, mask_zeros=mask_zeros, miss_area_names=self.get_miss_area_names())

        segmap = self.load_segmap(file_name)

        return file_name, self.image_transform(self.convert_pil(image)), self.mask_transform(self.convert_pil(mask)), \
               torch.FloatTensor(segmap), torch.FloatTensor(attr_matrix)


class CelebA_Face_Cls(data.Dataset):
    def __init__(self, image_dir, facial_attr_info_dict, is_train, image_transform):
        super(CelebA_Face_Cls, self).__init__()
        self.image_dir = image_dir
        self.is_train = is_train
        self.image_transform = image_transform

        self.dataset = facial_attr_info_dict['attr_dataset']
        self.dataset = self.dataset[1999:] if self.is_train else self.dataset[:1999]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        file_name, attr = self.dataset[item]
        image = Image.open(os.path.join(self.image_dir, file_name))
        # attr value
        attr = list(chain.from_iterable(attr))

        return file_name, self.image_transform(image), torch.FloatTensor(attr)


def generate_stroke_mask(mask_zeros, max_parts=9, maxVertex=25, maxLength=100, maxBrushWidth=24, maxAngle=360):
    parts = random.randint(1, max_parts)
    for i in range(parts):
        mask_zeros = mask_zeros + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle,
                                                    mask_zeros.shape[0], mask_zeros.shape[1])
    mask_zeros = np.minimum(mask_zeros, 1.0)
    return mask_zeros


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def get_dataloader(image_dir, mask_dir, facial_fea_names, facial_fea_attr_names,
                   facial_fea_attr_len, facial_attr_dataset, p_irregular_miss, max_num_miss, image_size=256,
                   dataset_name='CelebA_Attr_CV2', dilate_iter=2, is_train=True,
                   batch_size=2, num_workers=4):
    """return a data loader"""

    # transform
    image_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    if dataset_name == 'CelebA_Attr_CV2':
        dataset = CelebA_Attr_CV2(image_dir, mask_dir, image_size, facial_fea_names, facial_fea_attr_names,
                                  facial_fea_attr_len, facial_attr_dataset, p_irregular_miss, max_num_miss,
                                  dilate_iter, is_train, image_transform, mask_transform)
    elif dataset_name == 'CelebA_Face_Cls':
        dataset = None
    else:
        dataset = None

    if dataset:

        data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                      shuffle=is_train, num_workers=num_workers)
    else:
        raise ValueError('dataset {} do not support!'.format(dataset_name))

    return data_loader
