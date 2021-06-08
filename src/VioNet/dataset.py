import os
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from dynamic_image import dynamic_image_v1
from transformations.temporal_transforms import KeyFrameCrop, TrainGuidedKeyFrameCrop, ValGuidedKeyFrameCrop, KeySegmentCrop, SequentialCrop
from transformations.spatial_transforms import Lighting
from global_var import RGB_FRAME, DYN_IMAGE
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

def imread(path):
    with Image.open(path) as img:
        return img.convert('RGB')

def get_video_frames(video_path):
    # Initialize the frame number and create empty frame list
    video = cv2.VideoCapture(video_path)
    frame_list = []

    # Loop until there are no frames left.
    try:
        while True:
            more_frames, frame = video.read()

            if not more_frames:
                break
            else:
                frame_list.append(frame)

    finally:
        video.release()

    return frame_list



def video_loader(video_dir_path, frame_indices, dataset_name, input_type):
    video = []
    if input_type=='dynamic-images' and isinstance(frame_indices[0], list):
        for segment in frame_indices:
            shot_frames = []
            for i in segment:
                image_path = os.path.join(video_dir_path, 'frame{}.jpg'.format(i)) if dataset_name == 'rwf-2000' else os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
                # print('image_path:', image_path)
                if os.path.exists(image_path):
                    shot_frames.append(np.array(imread(image_path)))
            imgPIL, img = dynamic_image_v1(shot_frames)
            video.append(imgPIL)
        return video
    else:
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'frame{}.jpg'.format(i)) if dataset_name == 'rwf-2000' else os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
            # print('image_path:', image_path)
            if os.path.exists(image_path):
                video.append(imread(image_path))
        return video
    


def n_frames_loader(file_path):
    with open(file_path, 'r') as input_file:
        return float(input_file.read().rstrip('\n\r'))


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_labels(data):
    class_labels_map = {}
    # index = 0
    for class_label in data['labels']:
        index = 0 if class_label == "no" else 1
        class_labels_map[class_label] = index

        # class_labels_map[class_label] = index
        # index += 1
    print('class_labels_map:', class_labels_map)
    return class_labels_map


def get_video_names_and_labels(data, subset):
    video_names = []
    video_labels = []

    for key, val in data['database'].items():
        if val['subset'] == subset:
            label = val['annotations']['label']
            video_names.append(key)
            video_labels.append(label)

    return video_names, video_labels


def make_dataset(root_path, annotation_path, subset, dataset_name, tmp_annotation_path):
    """
    :param root_path: xxx
    :param annotation_path: xxx.json
    :param subset: 'train', 'validation', 'test'
    :return: list_of_videos, index_to_class_decode
    """

    data = load_annotation_data(annotation_path)

    video_names, video_labels = get_video_names_and_labels(data, subset)

    # print('video_names:', video_names)
    # print('video_labels:', video_labels)

    class_to_index = get_labels(data)
    index_to_class = {}
    for name, label in class_to_index.items():
        index_to_class[label] = name

    dataset = []
    if dataset_name == RWF_DATASET:
      for video_name, video_label in zip(video_names, video_labels):
        video_path = os.path.join(
            root_path, video_name
        )  # $1/$2/$3
        # print('video_path:', video_path)
        if not os.path.exists(video_path):
            continue

        # n_frames = int(n_frames_loader(os.path.join(video_path, 'n_frames')))
        # print('subset: ',subset)
        n_frames = len(os.listdir(video_path))
        # tmp_annotation = None if video_label == "no" and subset == "training" else os.path.join(tmp_annotation_path, video_name) + '.csv'
        tmp_annotation = os.path.join(tmp_annotation_path, video_name) + '.csv' if tmp_annotation_path else None

        video = {
            'name': video_name,
            'path': video_path,
            'label': class_to_index[video_label],
            'n_frames': n_frames,
            'tmp_annotation': tmp_annotation
        }

        dataset.append(video)
    else:
      for video_name, video_label in zip(video_names, video_labels):
          video_path = os.path.join(
              root_path, video_label, video_name
          )  # $1/$2/$3
          
          if not os.path.exists(video_path):
              continue

          n_frames = int(n_frames_loader(os.path.join(video_path, 'n_frames')))
          tmp_annotation = os.path.join(tmp_annotation_path, video_label, video_name) + '.csv' if tmp_annotation_path else None
          # print("tmp_annotation: {}, video_label:{}".format(tmp_annotation, video_label))
          video = {
              'name': video_name,
              'path': video_path,
              'label': class_to_index[video_label],
              'n_frames': n_frames,
              'tmp_annotation': tmp_annotation
          }

          dataset.append(video)
    # print('dataset: ', dataset)

    return dataset, index_to_class

"""

"""
class VioDB(Dataset):
    def __init__(
        self,
        root_path,
        annotation_path,
        subset,
        spatial_transform=None,
        temporal_transform=None,
        target_transform=None,
        dataset_name='',
        config=None,
        tmp_annotation_path=None,
        input_type='rgb'
    ):
        
        self.input_type = input_type
        self.videos, self.classes = make_dataset(
            root_path, annotation_path, subset, dataset_name, tmp_annotation_path
        )

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        self.loader = video_loader
        self.dataset_name = dataset_name

    def __getitem__(self, index):

        path = self.videos[index]['path']
        n_frames = self.videos[index]['n_frames']
        frames = list(range(1, 1 + n_frames))

        if self.temporal_transform:
            if isinstance(self.temporal_transform, KeyFrameCrop) or isinstance(self.temporal_transform, TrainGuidedKeyFrameCrop) or isinstance(self.temporal_transform, ValGuidedKeyFrameCrop):
                frames = self.temporal_transform(frames, self.videos[index]['tmp_annotation'])
            elif isinstance(self.temporal_transform, KeySegmentCrop):
                frames = self.temporal_transform(frames, self.videos[index]['tmp_annotation'])
            else:        
                frames = self.temporal_transform(frames)
                # print('frames: ', frames)

        clip = self.loader(path, frames, self.dataset_name, self.input_type)
        # print('clip type:', type(clip), len(clip), type(clip[0]))
       
        # clip list of images (H, W, C)
        if self.spatial_transform:
            clip = self.spatial_transform(clip)

        # clip: lists of tensors(C, H, W)
        clip = torch.stack(clip).permute(1, 0, 2, 3)

        target = self.videos[index]
        if self.target_transform:
            target = self.target_transform(target)
        
        

        return clip, target

    def __len__(self):
        return len(self.videos)

class ProtestDataset(Dataset):
    """
    dataset for training and evaluation
    """
    def __init__(self, txt_file, img_dir, transform = None, thr=0.5):
        """
        Args:
            txt_file: Path to txt file with annotation
            img_dir: Directory with images
            transform: Optional transform to be applied on a sample.
        """
        self.label_frame = pd.read_csv(txt_file, delimiter="\t").replace('-', 0)
        self.img_dir = img_dir
        self.transform = transform
        self.thr = thr

    def __len__(self):
        return len(self.label_frame)

    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_dir,
                                self.label_frame.iloc[idx, 0])
        image = imread(imgpath)
        fname = self.label_frame.iloc[idx]["fname"]
        violence = float(self.label_frame.iloc[idx]["violence"])
       
        label = 0 if violence<self.thr else 1

        # sample = {"image":image, "label":label}
        if self.transform:
            image = self.transform(image)
        return image, label

class ProtestDatasetEval(Dataset):
    """
    dataset for just calculating the output (does not need an annotation file)
    """
    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir: Directory with images
        """
        self.img_dir = img_dir
        self.transform = transform
        self.img_list = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_dir,
                                self.img_list[idx])
        image = imread(imgpath)
        if self.transform:
            image = self.transform(image)
        # we need this variable to check if the image is protest or not)
        # sample = {"imgpath":imgpath, "image":image}
        # sample["image"] = self.transform(sample["image"])
        return imgpath, image

class OneVideoFolderDataset(Dataset):
    """
    dataset for just calculating the output (does not need an annotation file)
    """
    def __init__(self, img_dir, dataset, out_type, spatial_transform=None, temporal_transform=None):
        """
        Args:
            img_dir: Directory with images
        """
        self.img_dir = img_dir
        self.spatial_transform = spatial_transform
        n_frames = len(os.listdir(img_dir))

        # print("vides list ({}/{}):".format(img_dir,n_frames), os.listdir(img_dir))
        
        frames = list(range(1, 1 + n_frames))
        self.segments = temporal_transform(frames)
        self.dataset=dataset
        self.out_type = out_type
        # self.template = 'frame{}.jpg' if dataset=="rwf-2000" else 'image_{}.jpg'
        # print("self.segments:",self.segments)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        segment_name = '-'.join([str(elem) for elem in segment])
        # print(idx,segment)
        images = []

        # real_images = []
        for frame_number in segment:
            if self.dataset == "rwf-2000":
                imgpath = os.path.join(self.img_dir, 'frame{}.jpg'.format(str(frame_number)))
            elif self.dataset == "hockey":
                imgpath = os.path.join(self.img_dir, 'image_{}.jpg'.format(str(frame_number).zfill(5)))
            # real_images.append(os.path.split(imgpath)[1])
            image = imread(imgpath)
            images.append(np.array(image))
          
        # print("real segment:",real_images)
        images = torch.from_numpy(np.stack(images,axis=0))
        if self.out_type == DYN_IMAGE:
            imgPIL, img = dynamic_image_v1(images)
            video = [imgPIL]
        elif self.out_type == RGB_FRAME:
            video = images #(T, H, W, C)
            # print("video: ", video.size())
        
        # images = images.permute(3,0,1,2)
        if self.spatial_transform:
            if isinstance(self.spatial_transform, tuple):
                video = self.spatial_transform[0](video)
                images = self.spatial_transform[1](images)
            else:
                # video = self.spatial_transform(video[0])
                # video = [video]
                video = self.spatial_transform(video)
        if self.out_type == DYN_IMAGE:
            video = torch.stack(video).permute(1, 0, 2, 3)
        
        return video, segment_name, images#(batch, c, T, w, h)

from global_var import *

if __name__ == "__main__":
    videos, classes = make_dataset(
            os.path.join(HOME_UBUNTU, RWF_DATASET.upper(),'frames/'),
            os.path.join(HOME_UBUNTU, VIO_DB_DATASETS, "rwf-2000_jpg1.json"),
            'validation',
            RWF_DATASET,
            os.path.join(HOME_UBUNTU, PATH_SCORES, "Scores-dataset(rwf-2000)-ANmodel(AnomalyDetector_Dataset(UCFCrime2LocalClips)_Features(c3d)_TotalEpochs(100000)_ExtraInfo(c3d)-Epoch-7000)-input(rgb)")
        )
    
    print('videos:', len(videos), videos[0:3])
    print('classes:', classes)

    # crop_method = GroupScaleCenterCrop(size=config.sample_size)
    # norm = Normalize([0.49778724, 0.49780366, 0.49776983], [0.09050678, 0.09017131, 0.0898702 ])
    # spatial_transform = Compose([crop_method, ToTensor(), norm])

    # mean = [0.49778724, 0.49780366, 0.49776983]
    # std = [0.09050678, 0.09017131, 0.0898702]
    # size = 224

    # spatial_transform = transforms.Compose([
    #         transforms.Resize(size),
    #         transforms.CenterCrop(size),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean, std)])

    # from video_transforms import ToTensorVideo, RandomResizedCropVideo, NormalizeVideo
    # mean = [124 / 255, 117 / 255, 104 / 255]
    # std = [1 / (.0167 * 255)] * 3
    # size = 112
    # spatial_transform = transforms.Compose([
    #     ToTensorVideo(),
    #     RandomResizedCropVideo(size, size),
    #     NormalizeVideo(mean=mean, std=std)
    # ])

    # temporal_transform = SequentialCrop(size=10, stride=1, overlap=0, max_segments=4)
    # val_dataset = OneVideoFolderDataset(img_dir="/Users/davidchoqueluqueroman/Documents/DATASETS_Local/VioDenseDatasets/hockey_jpg/fi/fi118_xvid",
    #                             dataset="hockey",
    #                             out_type=RGB_FRAME,
    #                             spatial_transform=spatial_transform, 
    #                             temporal_transform=temporal_transform)

    # video, segment_name, images = val_dataset[3]
    # print("video: ", video.size())
    # print("segment_name: ", segment_name)
    # print("images: ", images.size())
