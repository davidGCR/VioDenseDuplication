import os
import sys
sys.path.insert(1, '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/src')
sys.path.insert(1, '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/src/TubeletGeneration')

import re
import torch.utils.data as data
import torch
from torchvision import transforms
from TubeletGeneration.metrics import extract_tubes_from_video
from TubeletGeneration.tube_utils import JSON_2_videoDetections
from VioNet.customdatasets.make_dataset import MakeUCFCrime2LocalClips
from VioNet.dataset import video_loader
from VioNet.utils import natural_sort
from operator import itemgetter
from VioNet.customdatasets.dataset_utils import imread

class UCFCrime2LocalDataset(data.Dataset):
    """
    Load tubelets from one video
    Use to extract features tube-by-tube from just a video
    """

    def __init__(
        self, 
        root,
        path_annotations,
        abnormal,
        persons_detections_path,
        transform=None,
        clip_len=25,
        clip_temporal_stride=1):
        # self.dataset_root = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips',
        # self.split = 'anomaly',
        # self.video = 'Arrest036(2917-3426)',
        # self.p_d_path = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/ucfcrime2local',
        # self.gt_ann_path = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CrimeViolence2LocalDATASET/Txt annotations-longVideos'

        # self.dataset_root = dataset_root
        # self.split = split
        # self.video = video
        # self.p_d_path = p_d_path
        # self.gt_ann_path = gt_ann_path
        # self.transform = transform
        # self.person_detections = JSON_2_videoDetections(p_d_path)
        # self.tubes = extract_tubes_from_video(
        #     self.dataset_root,
        # )
        self.clip_temporal_stride = clip_temporal_stride
        self.clip_len = clip_len
        self.root = root
        self.path_annotations = path_annotations
        self.abnormal = abnormal
        self.make_dataset = MakeUCFCrime2LocalClips(root, path_annotations, abnormal)
        self.paths, self.labels, self.annotations = self.make_dataset()

        self.persons_detections_path = persons_detections_path

    def __len__(self):
        return len(self.paths)
    
    def get_video_clips(self, video_folder):
        _frames = os.listdir(video_folder)
        _frames = [f for f in _frames if '.jpg' in f]
        num_frames = len(_frames)
        indices = [x for x in range(0, num_frames, self.clip_temporal_stride)]
        indices_segments = [indices[x:x + self.clip_len] for x in range(0, len(indices), self.clip_len)]

        return indices_segments

    def generate_tube_proposals(self, path, frames):
        tmp = path.split('/')
        split = tmp[-2]
        video = tmp[-1]
        p_d_path = os.path.join(self.persons_detections_path, split, video)
        person_detections = JSON_2_videoDetections(p_d_path)
        tubes = extract_tubes_from_video(
            self.root,
            person_detections,
            frames,
            # {'wait': 200}
            )
        return tubes

    def __getitem__(self, index):
        path = self.paths[index]
        ann = self.annotations[index]
        sp_annotations_gt = self.make_dataset.ground_truth_boxes(path, ann)

        video_clips = self.get_video_clips(path)
        return video_clips, path, ann, sp_annotations_gt

class UCFCrime2LocalVideoDataset(data.Dataset):
    def __init__(
        self, 
        path,
        sp_annotation,
        p_detections,
        transform=None,
        clip_len=25,
        clip_temporal_stride=1):
        self.path = path
        self.sp_annotation = sp_annotation
        self.p_detections = p_detections
        self.transform = transform
        self.clip_len = clip_len
        self.clip_temporal_stride = clip_temporal_stride

        self.clips = self.get_video_clips(self.path)
        self.video_name = path.split('/')[-1]
        self.clase = path.split('/')[-2]
    
    def __len__(self):
        return len(self.clips)
    
    def split_list(self, lst, n):  
        for i in range(0, len(lst), n): 
            yield lst[i:i + n] 
    
    def get_video_clips(self, video_folder):
        _frames = os.listdir(video_folder)
        _frames = [f for f in _frames if '.jpg' in f]
        num_frames = len(_frames)

        # indices = [x for x in range(0, num_frames, self.clip_temporal_stride)]
        # indices_segments = [indices[x:x + self.clip_len] for x in range(0, len(indices), self.clip_len)]
        # real_clip_len = self.clip_len*self.clip_temporal_stride
        indices = [x for x in range(0, num_frames, self.clip_temporal_stride)]
        indices_segments = list(self.split_list(indices, self.clip_len)) 

        return indices_segments
    
    def load_frames(self, indices):
        image_names = os.listdir(self.path)
        image_names = natural_sort(image_names)
        image_names = list(itemgetter(*indices)(image_names))
        image_paths = [os.path.join(self.path,img_name) for img_name in image_names]
        images = []
        for ip in image_paths:
            img = self.transform(imread(ip)) if self.transform else imread(ip)
            images.append(img)
        print('len(images): ', len(images), type(images[0]))
        images = torch.stack(images, dim=0)
        return image_names, images
    
    def load_sp_annotations(self, frames, ann_path):
        frames_numbers = [int(re.findall(r'\d+', f)[0]) for f in frames]
        frames_numbers.sort()
        annotations = []
        with open(ann_path) as fid:
            lines = fid.readlines()
            ss = 1 if lines[0].split()[5] == '0' else 0
            for line in lines:
                # v_name = line.split()[0]
                # print(line.split())
                ann = line.split()
                frame_number = int(ann[5]) + ss
                valid = ann[6]
                if valid == '0' and frame_number in frames_numbers:
                    annotations.append(
                        {
                            "frame": frame_number,
                            "xmin": ann[1],
                            "ymin": ann[2],
                            "xmax": ann[3],
                            "ymax": ann[4]
                        }
                    )
        
        return annotations
    
    def __getitem__(self, index):
        clip = self.clips[index]
        image_names, images = self.load_frames(clip)
        gt = self.load_sp_annotations(image_names, self.sp_annotation)
        return clip, images, gt

if __name__=='__main__':
    # dataset = UCFCrime2LocalDataset(
    #     root='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips',
    #     path_annotations='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CrimeViolence2LocalDATASET/Txt annotations-longVideos',
    #     abnormal=True,
    #     persons_detections_path='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/ucfcrime2local',
    #     transform=None,
    #     clip_len=25,
    #     clip_temporal_stride=5)
    
    
    # video_clips, path, ann, sp_annotations_gt = dataset[45]
    # print(path)
    # print(ann)
    # print(video_clips)

    video_dataset = UCFCrime2LocalVideoDataset(
        path='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips/anomaly/Stealing091(245-468)',
        sp_annotation='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CrimeViolence2LocalDATASET/Txt annotations-longVideos/Stealing091.txt',
        p_detections='',
        transform=transforms.ToTensor(),
        clip_len=25,
        clip_temporal_stride=5
    )

    for clip, frames, gt in video_dataset:
        print('--',clip, len(clip), frames.size())
        for g in gt:
            print(g)