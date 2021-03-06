# 2021-12-3
# learning from https://github.com/stunback/DenseDepth-paddle/blob/main/data/data.py
# modified following "Deeper Depth Prediction with Fully Convolutional Residual Networks" Section 4.2

import os
import random
from io import BytesIO
import numpy as np
from zipfile import ZipFile
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import paddle
from paddle.io import Dataset
from paddle.vision import transforms

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}


class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth): raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth}


def loadZipToMem(zip_file):

    print('Loading dataset zip file...', end='')

    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list(
        (row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    nyu2_test = list(
        (row.split(',') for row in (data['data/nyu2_test.csv']).decode("utf-8").split('\n') if len(row) > 0))

    print('Loaded ({0}).'.format(len(nyu2_train)))
    return data, nyu2_train, nyu2_test

class ToTensor(object):

    def __init__(self, is_test=False):

        self.is_test = is_test

    def __call__(self, sample):
        
        image, depth = sample['image'], sample['depth']

        # resize
        image = image.resize((320, 240))
        depth = depth.resize((320, 240))

        image = np.array(image)/255.0
        image = image.astype(dtype=np.float32)
        depth = np.array(depth)
        depth = depth.reshape(240,320,1)
        depth = depth.astype(dtype=np.float32)

        if self.is_test:
            depth = depth/10.0
        else:
            depth = 1000.0*depth/255.0

        # normalize
        depth = depth.clip(10, 1000)
        # max depth is 10m
        depth = (depth/1000.0) * 10
        
        # crop center
        image = image[6:234,8:312,:]
        depth = depth[6:234,8:312,:]

        image = image.transpose(2,0,1)
        depth = depth.transpose(2,0,1)

        image = paddle.to_tensor(image)
        depth = paddle.to_tensor(depth)

        return {'image': image, 'depth': depth}

def getNoTransform(is_test=True):
    return transforms.Compose([
        ToTensor(is_test=is_test)
    ])


def getDefaultTrainTransform():
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor()
    ])


class NYUV2_Dataset(Dataset):

    def __init__(self, data, nyu2_train, transform=None):

        super(NYUV2_Dataset, self).__init__()
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform

    def __getitem__(self, idx):

        sample = self.nyu_dataset[idx]
        image = Image.open(BytesIO(self.data[sample[0]]))
        depth = Image.open(BytesIO(self.data[sample[1]]))
        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):

        return len(self.nyu_dataset)

def getTrainingTestingDataset(nyu_path):

    data, nyu2_train, nyu2_test = loadZipToMem(nyu_path)

    transformed_training = NYUV2_Dataset(data, nyu2_train, transform=getDefaultTrainTransform())
    transformed_testing = NYUV2_Dataset(data, nyu2_test, transform=getNoTransform())

    # sample = transformed_training[10]
    # print(sample['image'].shape)
    # print(sample['image'].numpy().max())
    # print(sample['depth'].shape)
    # print(sample['depth'].numpy().max())

    # sample = transformed_testing[10]
    # print(sample['image'].shape)
    # print(sample['image'].numpy().max())
    # print(sample['depth'].shape)
    # print(sample['depth'].numpy().max())

    return transformed_training, transformed_testing

if __name__ == '__main__':

    getTrainingTestingDataset('../nyu_data.zip')
