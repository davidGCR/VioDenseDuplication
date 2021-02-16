import os
import json
import torch
from torch.utils.data import Dataset

from PIL import Image
from dynamic_image import dynamic_image_v1
from temporal_transforms import KeyFrameCrop, TrainKeyFrameCrop, ValKeyFrameCrop
import numpy as np

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



def video_loader(video_dir_path, frame_indices, dataset_name):
    video = []
    if isinstance(frame_indices[0], list):
        for segment in frame_indices:
            shot_frames = []
            for i in segment:
                image_path = os.path.join(video_dir_path, 'frame{}.jpg'.format(i)) if dataset_name == 'rwf-2000' else os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
                # print('image_path:', image_path)
                if os.path.exists(image_path):
                    shot_frames.append(np.array(imread(image_path)))
            imgPIL, img = dynamic_image_v1(shot_frames)
            video.append(imgPIL)
            
    else:
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'frame{}.jpg'.format(i)) if dataset_name == 'rwf-2000' else os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
            # print('image_path:', image_path)
            if os.path.exists(image_path):
                video.append(imread(image_path))
            else:
                return video
    # else:
    #   for i in frame_indices:
    #       image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
    #       if os.path.exists(image_path):
    #           video.append(imread(image_path))
    #       else:
    #           return video

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
    if dataset_name == 'rwf-2000':
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
        tmp_annotation = None if video_label == "no" and subset == "training" else os.path.join(tmp_annotation_path, video_name) + '.csv'

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
          

          video = {
              'name': video_name,
              'path': video_path,
              'label': class_to_index[video_label],
              'n_frames': n_frames
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
        tmp_annotation_path=None
    ):
        

        self.videos, self.classes = make_dataset(
            root_path, annotation_path, subset, dataset_name, tmp_annotation_path
        )

        # print('self.videos: ', self.videos)
        # if config:

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        self.loader = video_loader
        self.dataset_name = dataset_name

    def __getitem__(self, index):

        path = self.videos[index]['path']
        n_frames = self.videos[index]['n_frames']
        frames = list(range(1, 1 + n_frames))

        # print('frames:', frames)

        if self.temporal_transform:
            if isinstance(self.temporal_transform, KeyFrameCrop) or isinstance(self.temporal_transform, TrainKeyFrameCrop) or isinstance(self.temporal_transform, ValKeyFrameCrop):
                frames = self.temporal_transform(frames, self.videos[index]['tmp_annotation'])
            else:        
                frames = self.temporal_transform(frames)

        # print('frames temporal_transform:', frames)

        clip = self.loader(path, frames, self.dataset_name)
        # print('path:',path)
        # print('clip:',len(clip))
        # print('clip:',len(clip))

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
