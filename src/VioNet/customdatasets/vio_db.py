import sys

import torchvision
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/VioDenseDuplication/src/VioNet')
import torch.utils.data as data
import os
from customdatasets.dataset_utils import imread
from transformations.dynamic_image_transformation import DynamicImage
import torch

TWO_STREAM_INPUT = {
    'input_1': {
        'type': 'rgb',
        'spatial_transform': None,
        'temporal_transform': None
    },
    'input_2': {
        'type': 'rgb',
        'spatial_transform': None,
        'temporal_transform': None
    }
}

class ViolenceDataset(data.Dataset):
    """
    
    """
    def __init__(self,
                       make_function,
                       dataset,
                       config
                       ):
        self.dataset = dataset
        self.config = config
        # self.input_type_1 = config['input_1']
        # self.spatial_transform = spatial_transform

        self.make_function = make_function
        self.paths, self.labels, self.annotations = self.make_function()
        self.max_video_len = 40 if dataset=='hockey' else 149
        # self.keyframe = keyframe
        # self.spatial_transform_2 = spatial_transform_2

        print('paths: {}, labels:{}, annot:{}'.format(len(self.paths), len(self.labels), len(self.annotations)))
        self.sampler = self.config['input_1']['temporal_transform']
        

    def load_input_1(self, path, seg):
        clip_images = []
        raw_clip_images = []
        if self.config['input_1']['type']=='rgb':
            frames_paths = [self.build_frame_name(path, i) for i in seg] #rwf
            for i in frames_paths:
                img = imread(i)
                clip_images.append(img)
            raw_clip_images = clip_images.copy()
            if self.config['input_1']['spatial_transform']:
                clip_images = self.config['input_1']['spatial_transform'](clip_images)
            
        return clip_images, raw_clip_images
    
    def build_frame_name(self, path, frame_number):
        if self.dataset == 'rwf-2000':
            return os.path.join(path,'frame{}.jpg'.format(frame_number+1))
        elif self.dataset == 'hockey':
            return os.path.join(path,'frame{:03}.jpg'.format(frame_number+1))

    
    def load_input_2(self, frames, path):
        if self.config['input_2']['type'] == 'rgb':
            i = frames[int(len(frames)/2)]
            img_path = self.build_frame_name(path, i)
            key_frame = imread(img_path)
            
        elif self.config['input_2']['type'] == 'dynamic-image':
            tt = DynamicImage()
            frames_paths = [self.build_frame_name(path, i) for i in frames] #rwf
            shot_images = [imread(img_path) for img_path in frames_paths]
            # img = self.spatial_transform(tt(shot_images)) if self.spatial_transform else tt(shot_images)
            key_frame = tt(shot_images)
        raw_key_frame = key_frame.copy()
        if self.config['input_2']['spatial_transform']:
            key_frame = self.config['input_2']['spatial_transform'](key_frame)
        return key_frame, raw_key_frame


    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        # annotation = self.annotations[index]
        frames = list(range(0, 150))
        frames = self.sampler(frames)
        # print(frames)

        video_images, raw_clip_images = self.load_input_1(path, frames)
        video_images = torch.stack(video_images, dim=0).permute(1,0,2,3)
        
        # raw_clip_images = 
        key_frame = None
        if self.config['input_2'] is not None:
            key_frame, raw_key_frame = self.load_input_2(frames, path)
            return video_images, label, path, key_frame, torchvision.transforms.ToTensor()(raw_key_frame)
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
