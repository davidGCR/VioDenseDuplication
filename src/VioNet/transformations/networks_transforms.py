# import sys
# import os
# g_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # print('main g_path:', g_path)
# sys.path.insert(1, g_path)

from transformations.video_transforms import ToTensorVideo, RandomResizedCropVideo, NormalizeVideo
from torchvision import transforms
import imgaug.augmenters as iaa
import numpy as np

class DIPredefinedTransforms(object):
    """
    Spatio-temporal transformation for train Dynamic Image Networks
    """
    def __init__(self, size, tmp_transform=None, mean=None, std=None):
        self.size = size
        self.mean = mean
        self.std = std
        self.tmp_transform = tmp_transform
        
        t = []
        # if tmp_transform:
        #     t.append(tmp_transform)
        self.train_transform = self.build_train_transform()
        self.val_transform = self.build_val_transform()

    def build_train_transform(self):
        t = [
                AfineTransformation(),
                transforms.Lambda(array2PIL),
                transforms.Resize(self.size),
                transforms.CenterCrop(self.size),
                transforms.ToTensor()
            ]

        if self.mean and self.std:
            t.append(transforms.Normalize(self.mean, self.std))
        t = transforms.Compose(t)
        return t

    def build_val_transform(self):
        t = [
                transforms.Resize(self.size),
                transforms.CenterCrop(self.size),
                transforms.ToTensor()
            ]
        if self.mean and self.std:
            t.append(transforms.Normalize(self.mean, self.std))
        t = transforms.Compose(t)
        return t

def s3d_transform(snippet):
    ''' stack & noralization '''
    # snippet = np.concatenate(snippet, axis=-1)
    # snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.float()
    snippet = snippet.mul_(2.).sub_(255).div(255)

    # out = snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)
    snippet = snippet.permute(3,0,1,2)

    return snippet
    
def c3d_fe_transform():
    mean = [124 / 255, 117 / 255, 104 / 255]
    std = [1 / (.0167 * 255)] * 3
    size = 112
    spatial_transform = transforms.Compose([
        ToTensorVideo(),
        RandomResizedCropVideo(size, size),
        NormalizeVideo(mean=mean, std=std)
    ])
    return spatial_transform



class AfineTransformation:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(1.00, [
                iaa.Affine(
                    scale={"x": (0.6, 1.4), "y": (0.6, 1.4)},
                    translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
                    rotate=(-15, 15),
                    shear=(-10, 10)
                ),
            ]),
            iaa.Fliplr(0.5),
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

def array2PIL(img):
    return Image.fromarray(img)

def dynamic_image_transform():
    mean=[0.49778724, 0.49780366, 0.49776983]
    std=[0.09050678, 0.09017131, 0.0898702]
    size = 224
    t = {
        'train': transforms.Compose([
                AfineTransformation(),
                transforms.Lambda(array2PIL),
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
        'val': transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
    }
    return t

def dynamic_image_transform_without_normalization():
    size = 224
    t=transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor()
            ])
    return t

if __name__== '__main__':
    t = dynamic_image_transform()
    print("transformation: ", type(t))