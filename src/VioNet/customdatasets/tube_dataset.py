import sys

from torch.utils.data.sampler import WeightedRandomSampler
sys.path.insert(1, '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/src/VioNet')

from numpy.core.numeric import indices
import torch.utils.data as data
import numpy as np
import torch
from torch.utils.data import dataset
import os
import random
from operator import itemgetter
# from MotionSegmentation.segmentation import segment
from customdatasets.dataset_utils import imread, filter_data_without_tubelet
from customdatasets.make_dataset import MakeRWF2000
from customdatasets.tube_crop import TubeCrop
from transformations.dynamic_image_transformation import DynamicImage


class TubeDataset(data.Dataset):
    """
    Load video tubelets
    Use this to load raw frames from tubes.
    """

    def __init__(self, frames_per_tube,
                       make_function,
                       max_num_tubes=4,
                       train=False,
                       dataset='',
                       random=True,
                       config=None):
        self.config = config
        self.dataset = dataset
        # self.input_type = input_type
        self.frames_per_tube = frames_per_tube
        # self.spatial_transform = spatial_transform
        self.make_function = make_function
        # if dataset == 'RealLifeViolenceDataset':
        #     self.paths, self.labels, self.annotations, self.num_frames = self.make_function()
        # else:
        #     self.paths, self.labels, self.annotations = self.make_function()
        self.paths, self.labels, self.annotations = self.make_function()
        self.paths, self.labels, self.annotations = filter_data_without_tubelet(self.paths, self.labels, self.annotations)

        # self.max_video_len = 39 if dataset=='hockey' else 149
        # self.keyframe = keyframe
        # self.spatial_transform_2 = spatial_transform_2

        print('paths: {}, labels:{}, annot:{}'.format(len(self.paths), len(self.labels), len(self.annotations)))
        self.sampler = TubeCrop(tube_len=frames_per_tube,
                                central_frame=True,
                                max_num_tubes=max_num_tubes,
                                train=train,
                                input_type=self.config['input_1']['type'],
                                # max_video_len=self.max_video_len,
                                random=random)
        self.max_num_tubes = max_num_tubes
    
    def get_sampler(self):
        class_sample_count = np.unique(self.labels, return_counts=True)[1]
        weight = 1./class_sample_count
        print('class_sample_count: ', class_sample_count)
        print('weight: ', weight)
        samples_weight = weight[self.labels]
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return sampler
    
    def build_frame_name(self, path, frame_number):
        if self.dataset == 'rwf-2000':
            return os.path.join(path,'frame{}.jpg'.format(frame_number+1))
        elif self.dataset == 'hockey':
            return os.path.join(path,'frame{:03}.jpg'.format(frame_number+1))
        elif self.dataset == 'RealLifeViolenceDataset':
            return os.path.join(path,'{:06}.jpg'.format(frame_number))
    
    def load_input_1(self, path, seg):
        tube_images = []
        raw_clip_images = []
        if self.config['input_1']['type']=='rgb':
            frames_paths = [self.build_frame_name(path, i) for i in seg] #rwf
            for i in frames_paths:
                img = imread(i)
                tube_images.append(img)
            raw_clip_images = tube_images.copy()
            if self.config['input_1']['spatial_transform']:
                tube_images = self.config['input_1']['spatial_transform'](tube_images)
        elif self.config['input_1']['type']=='dynamic-image':
            tt = DynamicImage()
            for shot in seg:
                frames_paths = [self.build_frame_name(path, i) for i in shot]
                shot_images = [imread(img_path) for img_path in frames_paths]
                img = self.spatial_transform(tt(shot_images)) if self.spatial_transform else tt(shot_images)
                tube_images.append(img)
        return tube_images, raw_clip_images
    
    def load_input_2(self, frames, path):
        if self.config['input_2']['type'] == 'rgb':
            i = frames[int(len(frames)/2)]
            # print('central frame:', i)
            img_path = self.build_frame_name(path, i)
            key_frame = imread(img_path)
            
        elif self.config['input_2']['type'] == 'dynamic-image':
            tt = DynamicImage()
            frames_paths = [self.build_frame_name(path, i) for i in frames] #rwf
            shot_images = [np.array(imread(img_path, resize=(224,224))) for img_path in frames_paths]
            # shot_images = [s.reshape((224,224,3)) for s in shot_images]
            # sizes = [s.shape for s in shot_images]
            # print('sizes: ', sizes)
            # img = self.spatial_transform(tt(shot_images)) if self.spatial_transform else tt(shot_images)
            key_frame = tt(shot_images)
            # print('key_frame: ', type(key_frame))
        raw_key_frame = key_frame.copy()
        if self.config['input_2']['spatial_transform']:
            key_frame = self.config['input_2']['spatial_transform'](key_frame)
        return key_frame, raw_key_frame

    def load_tube_images(self, path, seg):
        tube_images = [] #one tube-16 frames
        if self.input_type=='rgb':
            frames = [self.build_frame_name(path, i) for i in seg]
            for i in frames:
                img = imread(i)
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
        annotation = self.annotations[index]
        if self.dataset == 'RealLifeViolenceDataset':
            max_video_len = len(os.listdir(path)) - 1
        elif self.dataset=='hockey':
            max_video_len = 39
        elif self.dataset=='rwf-2000':
            max_video_len = 149
        boxes, segments, idxs = self.sampler(JSON_2_tube(annotation), annotation, max_video_len)
        # print('boxes: ', boxes, len(boxes))
        # print(path,' segments: ', segments, len(segments), max_video_len)

        video_images = []
        num_tubes = len(segments)
        for seg in segments:
            tube_images, _ = self.load_input_1(path, seg)
            video_images.append(torch.stack(tube_images, dim=0))
        
        key_frames = []
        if self.config['input_2'] is not None:
            for seg in segments:
                # i = seg[int(len(seg)/2)]
                # if self.dataset == 'rwf-2000':
                #     img_path = os.path.join(path,'frame{}.jpg'.format(i+1)) #rwf
                # elif self.dataset == 'hockey':
                #     img_path = os.path.join(path,'frame{:03}.jpg'.format(i+1))
                # key_frame = self.spatial_transform_2(imread(img_path)) if self.spatial_transform_2 else imread(img_path)
                key_frame, _ = self.load_input_2(seg, path)
                key_frames.append(key_frame)

        
        if len(video_images)<self.max_num_tubes:
            bbox_id = len(video_images)
            for i in range(self.max_num_tubes-len(video_images)):
                # video_images.append(torch.zeros_like(video_images[0]))
                video_images.append(video_images[len(video_images)-1])
                # boxes.append(torch.from_numpy(np.array([bbox_id+i, 0,0,223,223])).float().unsqueeze(0))
                p_box = boxes[len(boxes)-1]
                # p_box[0,0] = bbox_id+i
                boxes.append(p_box)
                if self.config['input_2'] is not None:
                    key_frames.append(key_frames[-1])
        for j,b in enumerate(boxes):
            b[0,0] = j
            # print(b)
        boxes = torch.stack(boxes, dim=0).squeeze()
        
        if len(boxes.shape)==1:
            boxes = torch.unsqueeze(boxes, dim=0)
        

        video_images = torch.stack(video_images, dim=0).permute(0,2,1,3,4)
        # return path, label, annotation, frames_names, boxes, video_images
        if self.config['input_2'] is not None:
            key_frames = torch.stack(key_frames, dim=0)
            if torch.isnan(key_frames).any().item():
                print('Detected Nan at: ', path)
            if torch.isinf(key_frames).any().item():
                print('Detected Inf at: ', path)
            # print('key_frames: ', key_frames.size())
            # print('video_images: ', video_images.size())
            return boxes, video_images, label, num_tubes, path, key_frames
        else:
            return boxes, video_images, label, num_tubes, path





from torch.utils.data.dataloader import default_collate

def my_collate_2(batch):
    # batch = filter (lambda x:x is not None, batch)
    boxes = [item[0] for item in batch if item[0] is not None]
    images = [item[1] for item in batch if item[1] is not None]
    labels = [item[2] for item in batch if item[2] is not None]
    num_tubes = [item[3] for item in batch if item[3] is not None]
    batch = (boxes, images, labels, num_tubes)
    return default_collate(batch)

def my_collate(batch):
    # print('BATCH: ', type(batch), len(batch), len(batch[0]))
    boxes = [item[0] for item in batch if item[0] is not None]
    images = [item[1] for item in batch if item[1] is not None]
    labels = [item[2] for item in batch if item[2] is not None]
    num_tubes = [item[3] for item in batch if item[3] is not None]
    paths = [item[4] for item in batch if item[4]]
    if len(batch[0]) == 6:
        key_frames = [item[5] for item in batch if item[5] is not None]
    # num_tubes = [batch[3][i] for i,item in enumerate(batch) if item[2] is not None]

    # print('BATCH filtered: ', len(boxes), len(images), len(labels), len(num_tubes), paths)
    # target = torch.LongTensor(target)

    # print('boxes:', type(boxes), len(boxes))
    # print('images:', type(boxes), len(images))
    # print('boxes[i]:', type(boxes[0]), boxes[0].size())
    # print('images[i]:', type(images[0]), images[0].size())
    # return [torch.stack(boxes, dim=0), torch.stack(images, dim=0)], labels
    boxes = torch.cat(boxes,dim=0)
    for i in range(boxes.size(0)):
        boxes[i][0] = i
        # print('-', i)

    images = torch.cat(images,dim=0)
    labels = torch.tensor(labels)
    num_tubes = torch.tensor(num_tubes)
    if len(batch[0]) == 6:
        key_frames = torch.cat(key_frames,dim=0)

        return boxes, images, labels, num_tubes, paths, key_frames#torch.stack(labels, dim=0)
    return boxes, images, labels, num_tubes, paths

class OneVideoTubeDataset(data.Dataset):
    """
    Load tubelets from one video
    Use to extract features tube-by-tube from just a video
    """

    def __init__(self, frames_per_tube,
                       video_path,
                       annotation_path,
                       spatial_transform=None,
                       max_num_tubes=0):
        self.frames_per_tube = frames_per_tube
        self.spatial_transform = spatial_transform
        self.video_path = video_path
        self.annotation_path = annotation_path
        self.max_num_tubes = max_num_tubes
        self.sampler = TubeCrop(tube_len=frames_per_tube, central_frame=True, max_num_tubes=max_num_tubes)
        self.boxes, self.segments, self.idxs = self.sampler(JSON_2_tube(annotation_path), annotation_path)
        print('self.idxs:', self.idxs)
    
    def __len__(self):
        if self.boxes is not None:
            return len(self.boxes)
        else:
            return 0

    def __getitem__(self, index):
        if self.boxes == None:
            return None, None
        frames_names = []
        segment = self.segments[index]
        box = self.boxes[index]
       
        frames = [os.path.join(self.video_path,'frame{}.jpg'.format(i+1)) for i in segment]
        frames_names.append(frames)
        tube_images = [] #one tube-16 frames
        for i in frames:
            img = self.spatial_transform(imread(i)) if self.spatial_transform else imread(i)
            tube_images.append(img)
        # video_images.append(torch.stack(tube_images, dim=0))
        tube_images = torch.stack(tube_images, dim=0) #torch.Size([16, 3, 224, 224])
        # print("tube_images stacked:", tube_images.size())
        # tube_images.permute(0,2,1,3,4)
        
        return box, tube_images

class TubeFeaturesDataset(data.Dataset):
    """
    Load tubelet features from files (txt)
    """

    def __init__(self, frames_per_tube,
                       min_frames_per_tube,
                       make_function,
                       max_num_tubes=4,
                       map_shape=(1,528,4,14,14)):
        # self.annotation_path = annotation_path
        self.max_num_tubes = max_num_tubes
        self.map_shape = map_shape
        self.paths, self.labels, self.annotations, self.feat_annotations = make_function()
        self.sampler = TubeCrop(tube_len=frames_per_tube, min_tube_len=min_frames_per_tube, central_frame=True, max_num_tubes=max_num_tubes)
        
    
    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        box_annotation = self.annotations[index]
        feat_annotation = self.feat_annotations[index]
        boxes, segments, idxs = self.sampler(JSON_2_tube(box_annotation), box_annotation)
        
        f_maps = self.read_features(feat_annotation, idxs)
        f_maps = torch.reshape(f_maps, self.map_shape)
        
        boxes = torch.stack(boxes, dim=0).squeeze()
        if len(boxes.shape)==1:
            boxes = torch.unsqueeze(boxes, dim=0)
        return boxes, f_maps, label
    
    # def get_feature(self):
    #     features = read_features(f"{feature_subpath}.txt", self.features_dim, self.bucket_size)
    
    def read_features(self, annotation_path, idxs, features_dim=413952):
        if not os.path.exists(annotation_path):
            raise Exception(f"Feature doesn't exist: {annotation_path}")
        features = None
        with open(annotation_path, 'r') as fp:
            data = fp.read().splitlines(keepends=False)
            # idxs = random.sample(range(len(data)), self.max_num_tubes)
            data = list(itemgetter(*idxs)(data))
            features = np.zeros((len(data), features_dim))
            for i, line in enumerate(data):
                features[i, :] = [float(x) for x in line.split(' ')]
        # features = features[0:max_features, :]
        features = torch.from_numpy(features).float()
        
        return features


import json
from torch.utils.data import DataLoader
from torchvision import transforms

def JSON_2_tube(json_file):
    """
    """
    with open(json_file, "r") as read_file:
        decodedArray = json.load(read_file)
        # print("decoded Array:", type(decodedArray), len(decodedArray))
        
        for f in decodedArray:
            for i, box in  enumerate(f['boxes']):
                f['boxes'][i] = np.asarray(f['boxes'][i])
        # print(decodedArray[0])
        decodedArray = sorted(decodedArray, key = lambda i: i['id'])
        return decodedArray

def check_no_tubes(make_function):
    paths, labels, annotations = make_function()
    videos_no_tubes = []
    for i, ann in enumerate(annotations):
        tubes = JSON_2_tube(ann)
        if len(tubes)==0:
            videos_no_tubes.append(paths[i])
    
    return videos_no_tubes


if __name__=='__main__':

    # tmp_crop = TubeCrop()
    # tubes = JSON_2_tube('/media/david/datos/Violence DATA/Tubes/RWF-2000/train/Fight/_6-B11R9FJM_0.json')
    # print('len tubes:', len(tubes), [(d['id'], d['len']) for d in tubes])
    # segments = tmp_crop(tubes)
    # print('segments: ', segments, len(segments))

    # make_dataset = MakeRWF2000(root='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames', 
    #                             train=True,
    #                             path_annotations='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/Tubes/RWF-2000')
    # dataset = TubeDataset(frames_per_tube=16, 
    #                         min_frames_per_tube=8, 
    #                         make_function=make_dataset,
    #                         spatial_transform=transforms.Compose([
    #                             transforms.CenterCrop(224),
    #                             transforms.ToTensor()
    #                         ]))
    # path, label, annotation,frames_names, boxes, video_images = dataset[213]
    # print('path: ', path)
    # print('label: ', label)
    # print('annotation: ', annotation)
    # print('boxes: ', boxes.size(), boxes)
    # print('video_images: ', video_images.size())
    # print('frames_names: ', frames_names)

    # loader = DataLoader(dataset,
    #                     batch_size=4,
    #                     shuffle=True,
    #                     num_workers=0,
    #                     pin_memory=True,
    #                     collate_fn=my_collate)

    # for i, data in enumerate(loader):
    #     # path, label, annotation,frames_names, boxes, video_images = data
    #     boxes, video_images, labels = data
    #     print('_____ {} ______'.format(i+1))
    #     # print('path: ', path)
    #     # print('label: ', label)
    #     # print('annotation: ', annotation)
    #     print('boxes: ', type(boxes), len(boxes), '-boxes[0]: ', boxes[0].size())
    #     print('video_images: ', type(video_images), len(video_images), '-video_images[0]: ', video_images[0].size())
    #     print('labels: ', type(labels), len(labels), '-labels: ', labels)

    crop = TubeCrop()
    lp = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    print('lp len:', len(lp))
    idxs = crop.__centered_frames__(lp)
    print('idxs: ', idxs, len(idxs))
