import os
import sys

import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision.datasets.video_utils import VideoClips
import torch
import torchvision
import glob
from operator import itemgetter

class VideoDataset(data.Dataset):
    """
    Process raw videos to get videoclips
    """
    def __init__(self,
                 clip_length,
                 frame_stride,
                 frame_rate=None,
                 dataset_path=None,
                 spatial_transform=None,
                 temporal_transform=None,
                 return_label=False,
                 video_formats=["avi", "mp4"]):
        super(VideoDataset, self).__init__()
        # video clip properties
        self.frames_stride = frame_stride
        self.total_clip_length_in_frames = clip_length * frame_stride
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.video_formats = video_formats
        # IO
        self.dataset_path = dataset_path
        self.video_list = self._get_video_list(dataset_path=self.dataset_path)
        # print("video_list:", self.video_list, len(self.video_list))
        self.return_label = return_label

        # data loading
        self.video_clips = VideoClips(video_paths=self.video_list,
                                      clip_length_in_frames=self.total_clip_length_in_frames,
                                      frames_between_clips=self.total_clip_length_in_frames,
                                      frame_rate=frame_rate)

    @property
    def video_count(self):
        return len(self.video_list)

    def getitem_from_raw_video(self, idx):
        video, _, _, _ = self.video_clips.get_clip(idx)
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)

        video_path = self.video_clips.video_paths[video_idx]

        in_clip_frames = list(range(0, self.total_clip_length_in_frames, self.frames_stride))

        # print("idx: {}, video_path: {}, video_idx: {}, clip_idx: {}, in_clip_frames: {}".format(idx, video_path, video_idx, clip_idx, in_clip_frames))

        video = video[in_clip_frames]
        if self.temporal_transform:
            video = self.temporal_transform(video)
        
        if self.spatial_transform:
            video = self.spatial_transform(video)

        dir, file = video_path.split(os.sep)[-2:]
        file = file.split('.')[0]

        # if self.return_label:
        #     label = 0 if "Normal" in video_path else 1
        #     return video, label, clip_idx, dir, file
        label = 0 if "Normal" in video_path else 1

        return video, label, (clip_idx, dir, file)

    def __len__(self):
        return len(self.video_clips)

    def __getitem__(self, index):
        succ = False
        while not succ:
            try:
                batch = self.getitem_from_raw_video(index)
                succ = True
            except Exception as e:
                index = np.random.choice(range(0, self.__len__()))
                trace_back = sys.exc_info()[2]
                line = trace_back.tb_lineno
                logging.warning(f"VideoIter:: ERROR (line number {line}) !! (Force using another index:\n{index})\n{e}")

        return batch

    def _get_video_list(self, dataset_path):
        assert os.path.exists(dataset_path), "VideoIter:: failed to locate: `{}'".format(dataset_path)
        vid_list = []
        for path, subdirs, files in os.walk(dataset_path):
            for name in files:
                if not any([format in name and name[0]!= '.' for format in self.video_formats]):
                    continue
                vid_list.append(os.path.join(path, name))
        return vid_list

class HMDB51DatasetV2(data.Dataset):
    """
    Encapsules the hmdb51 original pytorch dataset to return just video clip and label
    """
    def __init__(self, root, annotation_path, frames_per_clip, step_between_clips, fold, train, transform):
        self.dataset = torchvision.datasets.HMDB51(root=root,
                                            annotation_path=annotation_path,
                                            frames_per_clip=frames_per_clip,
                                            step_between_clips=step_between_clips,
                                            fold=fold,
                                            train=train,
                                            transform=transform)
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        v, a, l = self.dataset[idx]
        return v, l

g_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, g_path)
from transformations.temporal_transforms import DynamicImage, tensor2PIL, PIL2numpy
from transformations.spatial_transforms import AfineTransformation, array2PIL, DIPredefinedTransforms
from utils import show_batch
import matplotlib.pyplot as plt

if __name__=='__main__':

    DN = DynamicImage(output_type="pil")
    di_t = DIPredefinedTransforms(size=224, tmp_transform=DN)
    ST = di_t.val_transform
    # ST = torchvision.transforms.Compose([
    #     DynamicImage(output_type="pil"),
    #     AfineTransformation(),
    #     torchvision.transforms.Lambda(array2PIL), #lambda x: Image.fromarray(x)
    #     torchvision.transforms.Resize(224),
    #     torchvision.transforms.CenterCrop(224),
    #     torchvision.transforms.ToTensor()
    # ])
    dataset = VideoDataset(clip_length=10,
                        frame_stride=1,
                        frame_rate=25,
                        dataset_path= "/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/Anomaly-Videos-All",#"/Users/davidchoqueluqueroman/Documents/DATASETS_Local/hmdb51/hmdb51_org",#"/Volumes/TOSHIBA EXT/DATASET/HockeyFight/videos",
                        temporal_transform=DN,
                        spatial_transform=ST)#"/Users/davidchoqueluqueroman/Documents/CODIGOS/DATASETS/UCFCrime/Abuse")
    
   

    batch=[]
    for i in range(6):
        video, clip_idx, dir, file = dataset[i]
        print("video:", type(video), video.dtype, video.size())
        print("dir:",dir)
        print("file:", file)
        print("clip_idx:", clip_idx)
        batch.append(video)
    batch = torch.stack(batch, dim=0)
    print("batch: ", batch.size())
    grid = torchvision.utils.make_grid(batch, nrow=6, padding=50)
    show_batch(grid)
    plt.show()

    # hmdb51_data_val = HMDB51DatasetV2(root='/Users/davidchoqueluqueroman/Documents/CODIGOS/DATASETS_Local/hmdb51/hmdb51_org',
    #                                           annotation_path='/Users/davidchoqueluqueroman/Documents/CODIGOS/DATASETS_Local/hmdb51/testTrainMulti_7030_splits',
    #                                           frames_per_clip=10,
    #                                           step_between_clips=100,
    #                                           fold=1,
    #                                           train=False,
    #                                           transform=ST)                                              
    # data_loader = torch.utils.data.DataLoader(hmdb51_data_val,
    #                                         batch_size=8,
    #                                         shuffle=True,
    #                                         num_workers=4)

    # print('Train dataset:', len(hmdb51_data_train))
    # print('Val dataset:', len(hmdb51_data_val))

    # v, a, l = hmdb51_data_val[5000]
    # print("video:", v.size())
    # print("label:", l)
    # grid = torchvision.utils.make_grid(v.permute(0,3,1,2), nrow=6, padding=50)
    # show(grid)
    # plt.show()

    # di = DN(v)
    # di.show()

    # for idx, (v, l) in enumerate(data_loader):
    #     if idx > 5:
    #         break
    #     print("video:", v.size(), v.dtype)
    #     print("label:", l.size())
        # print("video:", v.shape)
        # grid = torchvision.utils.make_grid(v, nrow=5, padding=20)
        # show(grid)
        # for v in batch:
        #     # v = ST(v)
        #     grid = torchvision.utils.make_grid(v.permute(0,3,1,2), nrow=5, padding=20)
        #     show(grid)
        #     plt.show()
        #     di = DN(v)
        #     di.show()