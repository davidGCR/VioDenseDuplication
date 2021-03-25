import os
from temporal_transforms import SegmentsCrop
import matplotlib.pyplot as plt
import matplotlib as mpl
from make_dataset import MakeImageHMDB51
from temporal_transforms import SegmentsCrop, DynamicImage
from utils import show_batch
import torch.utils.data as data
import torchvision
from operator import itemgetter
import torch

class VideoImageDataset(data.Dataset):
    """
    Load videos stored as images folders
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
                       spatial_transform=None):
        self.root = root
        self.frames_per_clip = frames_per_clip
        self.number_of_clips = number_of_clips
        self.temporal_transform = temporal_transform
        self.spatial_transform = spatial_transform
        self.paths, self.labels = make_function()
        self.sampler = SegmentsCrop(size=self.number_of_clips, segment_size=self.frames_per_clip, stride=stride, overlap=overlap, padding=padding, position=position)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        video_path = self.paths[idx]
        video_label = self.labels[idx]
        frames_paths = os.listdir(video_path)
        frames_paths.sort()
        frames_paths = [os.path.join(video_path,p) for p in frames_paths]
        # print(frames_paths)
        frames_idx = range(len(frames_paths))        
        segments = self.sampler(frames_idx)
        # print("segmenets:", segments)
        dynamic_images = []
        for s in segments:
            s = list(itemgetter(*s)(frames_paths)) #get frames paths
            image = self.temporal_transform(s)
            if self.spatial_transform:
                image = self.spatial_transform(image)
            dynamic_images.append(image)
        dynamic_images = torch.stack(dynamic_images, dim=0)
        return dynamic_images, video_label

if __name__=='__main__':
    m = MakeImageHMDB51(root="/Users/davidchoqueluqueroman/Documents/CODIGOS/DATASETS_Local/hmdb51/frames",
                        annotation_path="/Users/davidchoqueluqueroman/Documents/CODIGOS/DATASETS_Local/hmdb51/testTrainMulti_7030_splits",
                        fold=1,
                        train=True)
    # paths, labels = m()
    # print(paths[23], labels[23])
    temporal_transform = DynamicImage(output_type="pil")

    from spatial_transforms import DIPredefinedTransforms
    # mean=None
    # std=None
    mean = [0.49778724, 0.49780366, 0.49776983]
    std = [0.09050678, 0.09017131, 0.0898702]
    spatial_transform = DIPredefinedTransforms(size=224, tmp_transform=None, mean=mean, std=std)
    d=VideoImageDataset(root="",
                        frames_per_clip=10, 
                        number_of_clips=5, 
                        make_function=m, 
                        stride=1, 
                        overlap=0,
                        position="start",
                        temporal_transform=temporal_transform, 
                        spatial_transform=spatial_transform.val_transform)
    import random
    for i in random.sample(range(1,1000),10):
        print(i)
        v, l = d[i]
        print("video:", v.size(), "label: ", l)
        grid = torchvision.utils.make_grid(v, nrow=6, padding=50)
        show_batch(grid)
        plt.show()
