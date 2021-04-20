import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.utils.data as data
import torchvision
from operator import itemgetter
import torch
import re


g_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print('main g_path:', g_path)
sys.path.insert(1, g_path)
from customdatasets.make_dataset import MakeImageHMDB51, MakeRWF2000, MakeUCFCrime2LocalClips
from transformations.temporal_transforms import SegmentsCrop, DynamicImage
from transformations.temporal_transforms import SegmentsCrop
from transformations.spatial_transforms import DIPredefinedTransforms
from utils import show_batch

class VideoImageDataset(data.Dataset):
    """
    Load videos stored as images folders.
    Use this to load dinamic images. 
    It has embeded the SEGMENTS CROP temporal transformation to split a 
    video in video segments.
    """

    def __init__(self, root, 
                       frames_per_clip,
                       number_of_clips,
                       stride,
                       overlap,
                       make_function,
                       position,
                       padding=True,
                       temporal_transform=None,
                       spatial_transform=None,
                       return_metadata=False):
        self.root = root
        self.frames_per_clip = frames_per_clip
        self.number_of_clips = number_of_clips
        self.temporal_transform = temporal_transform
        self.spatial_transform = spatial_transform
        self.make_function = make_function
        self.paths, self.labels = self.make_function()
        self.sampler = SegmentsCrop(size=self.number_of_clips, segment_size=self.frames_per_clip, stride=stride, overlap=overlap, padding=padding, position=position)
        self.return_metadata = return_metadata
    
    def video_count(self):
        return len(self.paths)

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        video_path = self.paths[idx]
        video_label = self.labels[idx]
        frames_paths = os.listdir(video_path)
        frames_paths = [path for path in frames_paths if path[0]!='.']
        frames_paths.sort(key=lambda f: int(re.sub('\D', '', f)))

        frames_paths = [os.path.join(video_path,p) for p in frames_paths]
        # print(frames_paths)
        frames_idx = range(len(frames_paths))        
        segments = self.sampler(frames_idx)
        # print("video:", video_path, video_label)
        

        dynamic_images = []
        for i,s in enumerate(segments):
            s = list(itemgetter(*s)(frames_paths)) #get frames paths
            # print("segmenets-{}:".format(i+1), [os.path.split(path)[1] for path in s])
            image = self.temporal_transform(s)
            if self.spatial_transform:
                image = self.spatial_transform(image)
            dynamic_images.append(image)
        dynamic_images = torch.stack(dynamic_images, dim=0)

        if self.return_metadata:
            # clip_idx = 
            # _,file = os.path.split(video_path)
            # dir = self.make_function.classes[video_label]
            return dynamic_images, video_label, video_path ##TO DO
        else:    
            return dynamic_images, video_label

if __name__=='__main__':
    # m = MakeImageHMDB51(root="/Users/davidchoqueluqueroman/Documents/CODIGOS/DATASETS_Local/hmdb51/frames",
    #                     annotation_path="/Users/davidchoqueluqueroman/Documents/CODIGOS/DATASETS_Local/hmdb51/testTrainMulti_7030_splits",
    #                     fold=1,
    #                     train=True)
    # m = MakeRWF2000(root="/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames", train=True)
    # paths, labels = m()
    # print(paths[23], labels[23])
    temporal_transform = DynamicImage(output_type="pil")
    m = MakeUCFCrime2LocalClips(root=('/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips',
                                 '/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/frames'))
    
    # mean=None
    # std=None
    mean = [0.49778724, 0.49780366, 0.49776983]
    std = [0.09050678, 0.09017131, 0.0898702]
    spatial_transform = DIPredefinedTransforms(size=224, tmp_transform=None, mean=None, std=None)
    d=VideoImageDataset(root="",
                        frames_per_clip=10, 
                        number_of_clips=0, 
                        make_function=m, 
                        stride=1, 
                        overlap=0,
                        position="",
                        padding=False,
                        return_metadata=True,
                        temporal_transform=temporal_transform, 
                        spatial_transform=spatial_transform.val_transform) #spatial_transform.train_transform
    # import random
    # v, l, path = d[random.randint(0,len(d)-1)]
    # print("video:", path, v.size(), "label: ", l)
    # grid = torchvision.utils.make_grid(v, nrow=10, padding=50)
    # show_batch(grid)
    # plt.show()

    data_iter = torch.utils.data.DataLoader(d,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1,
                                            pin_memory=True)
    for v, l, path in data_iter:
        v = torch.squeeze(v)
        print("video:", path, v.size(), "label: ", l)
        grid = torchvision.utils.make_grid(v, nrow=10, padding=50)
        show_batch(grid, title=os.path.split(path[0])[1])
        plt.show()

    # for i in random.sample(range(1,100),10):
    #     print(i)
    #     v, l = d[i]
    #     print("video:", v.size(), "label: ", l)
    #     grid = torchvision.utils.make_grid(v, nrow=6, padding=50)
    #     show_batch(grid)
    #     plt.show()
