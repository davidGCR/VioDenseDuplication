import sys

import torchvision
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/VioDenseDuplication/src/VioNet')
import torch.utils.data as data
import os
from customdatasets.dataset_utils import imread
from transformations.dynamic_image_transformation import DynamicImage
import torch

class ViolenceDataset(data.Dataset):
    """
    
    """
    def __init__(self, temp_crop,
                       make_function,
                       spatial_transform=None,
                       train=False,
                       dataset='',
                       input_type='rgb',
                       random=True,
                       keyframe=False,
                       spatial_transform_2=None):
        self.dataset = dataset
        self.input_type = input_type
        self.spatial_transform = spatial_transform
        self.make_function = make_function
        self.paths, self.labels, self.annotations = self.make_function()
        self.max_video_len = 40 if dataset=='hockey' else 149
        self.keyframe = keyframe
        self.spatial_transform_2 = spatial_transform_2

        print('paths: {}, labels:{}, annot:{}'.format(len(self.paths), len(self.labels), len(self.annotations)))
        self.sampler = temp_crop
    
    def load_images(self, path, seg):
        tube_images = [] #one tube-16 frames
        if self.input_type=='rgb':
            if self.dataset == 'rwf-2000':
                frames = [os.path.join(path,'frame{}.jpg'.format(i+1)) for i in seg] #rwf
            elif self.dataset == 'hockey':
                frames = [os.path.join(path,'frame{:03}.jpg'.format(i+1)) for i in seg]
            for i in frames:
                img = self.spatial_transform(imread(i)) if self.spatial_transform else imread(i)
                tube_images.append(img)
        else:
            tt = DynamicImage()
            for shot in seg:
                if self.dataset == 'rwf-2000':
                    frames = [os.path.join(path,'frame{}.jpg'.format(i+1)) for i in shot] #rwf
                elif self.dataset == 'hockey':
                    frames = [os.path.join(path,'frame{:03}.jpg'.format(i+1)) for i in shot]
                shot_images = [imread(img_path) for img_path in frames]
                img = self.spatial_transform(tt(shot_images)) if self.spatial_transform else tt(shot_images)
                tube_images.append(img)
        return tube_images

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        # annotation = self.annotations[index]
        frames = list(range(0, 150))
        frames = self.sampler(frames)
        # print(frames)

        video_images = self.load_images(path, frames)
        video_images = torch.stack(video_images, dim=0).permute(1,0,2,3)
        
        key_frame = None
        if self.keyframe:
            i = frames[int(len(frames)/2)]
            # print('keyframe: ', i)
            if self.dataset == 'rwf-2000':
                img_path = os.path.join(path,'frame{}.jpg'.format(i+1)) #rwf
            elif self.dataset == 'hockey':
                img_path = os.path.join(path,'frame{:03}.jpg'.format(i+1))
            key_frame = self.spatial_transform_2(imread(img_path)) if self.spatial_transform_2 else imread(img_path)
        
        if self.keyframe:
            return video_images, label, path, key_frame
        else:
            return video_images, label, path

from transformations.temporal_transforms import RandomCrop, CenterCrop
from customdatasets.make_dataset import MakeRWF2000
from global_var import *
from config import Config
from torch.utils.data import DataLoader


if __name__=='__main__':
    # temporal_transform = RandomCrop(size=16, stride=1, input_type='rgb')
    temporal_transform = CenterCrop(size=16, stride=1, input_type='rgb')
    make_dataset = MakeRWF2000(
        root=os.path.join(HOME_UBUNTU, 'RWF-2000/frames'),#'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames', 
                        train=True,
                        path_annotations=os.path.join(HOME_UBUNTU, 'ActionTubes/RWF-2000'),
                        shuffle=False)
    dataset = ViolenceDataset(
        temporal_transform,
        make_dataset,
        dataset='rwf-2000',
        spatial_transform=torchvision.transforms.ToTensor(),
        keyframe=True,
        spatial_transform_2=torchvision.transforms.ToTensor(),

    )

    for i in range(len(dataset)):
        video_images, label, path, key_frame = dataset[i]
        print('video_images: ', video_images.size())
        print('label: ',label)
        print('key_frame: ', key_frame.size())
