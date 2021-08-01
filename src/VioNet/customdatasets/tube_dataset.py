import sys
sys.path.insert(1, '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/src/VioNet')

from numpy.core.numeric import indices
import torch.utils.data as data
import numpy as np
import torch
from torch.utils.data import dataset
from operator import itemgetter
import os
import random
from operator import itemgetter
# from MotionSegmentation.segmentation import segment
from customdatasets.dataset_utils import imread
from customdatasets.make_dataset import MakeRWF2000
from transformations.dynamic_image_transformation import DynamicImage

class OneVideoTubeDataset(data.Dataset):
    """
    Load tubelets from one video
    Use to extract features tube-by-tube from just a video
    """

    def __init__(self, frames_per_tube,
                       min_frames_per_tube,
                       video_path,
                       annotation_path,
                       spatial_transform=None,
                       max_num_tubes=0):
        self.frames_per_tube = frames_per_tube
        self.min_frames_per_tube = min_frames_per_tube
        self.spatial_transform = spatial_transform
        self.video_path = video_path
        self.annotation_path = annotation_path
        self.max_num_tubes = max_num_tubes
        self.sampler = TubeCrop(tube_len=frames_per_tube, min_tube_len=min_frames_per_tube, central_frame=True, max_num_tubes=max_num_tubes)
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

class TubeDataset(data.Dataset):
    """
    Load video tubelets
    Use this to load raw frames from tubes.
    """

    def __init__(self, frames_per_tube,
                       min_frames_per_tube,
                       make_function,
                       spatial_transform=None,
                       return_metadata=False,
                       max_num_tubes=16,
                       train=False,
                       dataset='',
                       input_type='rgb',
                       random=True):
        self.dataset = dataset
        self.input_type = input_type
        self.frames_per_tube = frames_per_tube
        self.min_frames_per_tube = min_frames_per_tube
        self.spatial_transform = spatial_transform
        self.make_function = make_function
        self.paths, self.labels, self.annotations = self.make_function()
        self.paths, self.labels, self.annotations = self.filter_data_without_tubelet()

        self.max_video_len = 40 if dataset=='hockey' else 149

        print('paths: {}, labels:{}, annot:{}'.format(len(self.paths), len(self.labels), len(self.annotations)))
        self.sampler = TubeCrop(tube_len=frames_per_tube, 
                                min_tube_len=min_frames_per_tube, 
                                central_frame=True,
                                max_num_tubes=max_num_tubes,
                                train=train,
                                input_type=input_type,
                                max_video_len=self.max_video_len,
                                random=random)
        self.return_metadata = return_metadata
        self.max_num_tubes = max_num_tubes
    
    def filter_data_without_tubelet(self):
        indices_2_remove = []
        for index in range(len(self.paths)):
            path = self.paths[index]
            label = self.labels[index]
            annotation = self.annotations[index]
            tubelets = JSON_2_tube(annotation)
            if len(tubelets) == 0:
                # print('No tubelets at: ',path)
                indices_2_remove.append(index)

        paths = [self.paths[i] for i in range(len(self.paths)) if i not in indices_2_remove]
        labels = [self.labels[i] for i in range(len(self.labels)) if i not in indices_2_remove]
        annotations = [self.annotations[i] for i in range(len(self.annotations)) if i not in indices_2_remove]
        return paths, labels, annotations
    
    def load_tube_images(self, path, seg):
        tube_images = [] #one tube-16 frames
        if self.input_type=='rgb':
            if self.dataset == 'rwf-2000':
                # print(seg)
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
        annotation = self.annotations[index]
        boxes, segments, idxs = self.sampler(JSON_2_tube(annotation), annotation)

        if boxes == None or len(boxes) == 0:
            return None, None, None, None, None
        video_images = []
        num_tubes = len(segments)
        for seg in segments:
            # print('seg:',seg)
            tube_images = self.load_tube_images(path, seg)
            video_images.append(torch.stack(tube_images, dim=0))
        
        if len(video_images)<self.max_num_tubes:
            bbox_id = len(video_images)
            for i in range(self.max_num_tubes-len(video_images)):
                # video_images.append(torch.zeros_like(video_images[0]))
                video_images.append(video_images[len(video_images)-1])
                # boxes.append(torch.from_numpy(np.array([bbox_id+i, 0,0,223,223])).float().unsqueeze(0))
                p_box = boxes[len(boxes)-1]
                # p_box[0,0] = bbox_id+i
                boxes.append(p_box)
        for j,b in enumerate(boxes):
            b[0,0] = j
            # print(b)
        boxes = torch.stack(boxes, dim=0).squeeze()
        
        if len(boxes.shape)==1:
            boxes = torch.unsqueeze(boxes, dim=0)
        

        video_images = torch.stack(video_images, dim=0).permute(0,2,1,3,4)
        # return path, label, annotation, frames_names, boxes, video_images
        return boxes, video_images, label, num_tubes, path



class TubeCrop(object):
    def __init__(self, 
                    tube_len=16, 
                    min_tube_len=8, 
                    central_frame=True, 
                    max_num_tubes=4, 
                    train=True,
                    input_type='rgb',
                    max_video_len=40,
                    random=True):
        """
        Args:
        """
        self.tube_len = tube_len
        self.min_tube_len = min_tube_len
        self.central_frame = central_frame
        self.max_num_tubes = max_num_tubes
        self.train = train
        self.input_type = input_type
        self.max_video_len = max_video_len
        self.random = random

    def __call__(self, tubes: list, tube_path: str):
        # assert len(tubes) >= 1, "No tubes in video!!!==>{}".format(tube_path)
        segments = []
        boxes = []
        for tube in tubes:
            if self.input_type=='rgb':
                tmp = tube['foundAt'].copy()
                frames_idxs = self.__centered_frames__(tube['foundAt'])
                # if len(tmp) < self.tube_len:
                #     print('very short tube: ', tube_path, frames_idxs, 'foundAt: ', tube['foundAt'])
            else:
                frames_idxs = self.__centered_segments__()
            if len(frames_idxs) > 0:
                bbox = self.__central_bbox__(tube['boxes'], tube['id']+1)
                boxes.append(bbox)
                segments.append(frames_idxs)
        # print('segments: ', segments,len(segments))
        idxs = range(len(boxes))
        if self.max_num_tubes != 0 and len(boxes) > self.max_num_tubes:
            if self.random:
                if self.train:
                    idxs = random.sample(range(len(boxes)), self.max_num_tubes)
                    boxes = list(itemgetter(*idxs)(boxes))
                    segments = list(itemgetter(*idxs)(segments))
                else:
                    n = len(boxes)
                    m = int(n/2)
                    # arr = np.array(boxes)
                    boxes = boxes[m-int(self.max_num_tubes/2) : m+int(self.max_num_tubes/2)]
                    segments = segments[m-int(self.max_num_tubes/2) : m+int(self.max_num_tubes/2)]
                    # boxes = boxes.tolist()
                    # segments = segments.tolist()
            else:
                tubes = sorted(tubes, key = lambda i: i['score'], reverse=True)
                boxes = boxes[0:self.max_num_tubes]
                segments = segments[0:self.max_num_tubes]
        for id,box in enumerate(boxes):
            boxes[id][0,0] = id

        if len(boxes) == 0:
            return None, None, None
        
        return boxes, segments, idxs
    
    def __centered_frames__(self, tube_frames_idxs: list):
        if len(tube_frames_idxs) == self.tube_len: 
            return tube_frames_idxs
        if len(tube_frames_idxs) > self.tube_len:
            n = len(tube_frames_idxs)
            m = int(n/2)
            arr = np.array(tube_frames_idxs)
            centered_array = arr[m-int(self.tube_len/2) : m+int(self.tube_len/2)]
            return centered_array.tolist()
        if len(tube_frames_idxs) < self.tube_len: #padding

            # last_idx = tube_frames_idxs[-1]
            # tube_frames_idxs += (self.tube_len - len(tube_frames_idxs))*[last_idx]
            # tube_frames_idxs = tube_frames_idxs
            # print('len(tube_frames_idxs) < self.tube_len: ', tube_frames_idxs)
            center_idx = int(len(tube_frames_idxs)/2)
            
            # print('center_idx:{}, center_frame:{}'.format(center_idx, tube_frames_idxs[center_idx]))
            
            start = tube_frames_idxs[center_idx]-int(self.tube_len/2)
            end = tube_frames_idxs[center_idx]+int(self.tube_len/2)
            # print('start: {}, end: {}'.format(start,end))
            out = list(range(start,end))
            # if tube_frames_idxs[center_idx]-int(self.tube_len/2) < self.max_video_len:
            if out[0]<0:
                most_neg = abs(out[0])
                out = [i+most_neg for i in out]
            elif tube_frames_idxs[center_idx]+int(self.tube_len/2) > self.max_video_len:
                start = tube_frames_idxs[center_idx]-(self.tube_len-(self.max_video_len-tube_frames_idxs[center_idx]))+1
                end = self.max_video_len+1
                out = list(range(start,end))
            tube_frames_idxs = out
            return tube_frames_idxs
    
    def __centered_segments__(self):
        """
        the overlap could cause less than 16 frames per tube
        """
        segment_size = 5
        stride = 1
        overlap = 0.5
        overlap_length = int(overlap*segment_size)
        total_len = self.tube_len*(segment_size-overlap_length)

        indices = [x for x in range(0, total_len, stride)]
        indices_segments = [indices[x:x + segment_size] for x in range(0, len(indices), segment_size-overlap_length)]
        indices_segments = [s for s in indices_segments if len(s)==segment_size]

        m = 75
        frames = list(range(150))
        frames = frames[m-int(total_len/2) : m+int(total_len/2)]
        video_segments = []
        for i, indices_segment in enumerate(indices_segments): #Generate segments using indices
            segment = np.asarray(frames)[indices_segment].tolist()
            video_segments.append(segment)
        return video_segments

    def __central_bbox__(self, tube, id):
        width, height = 224, 224
        if len(tube)>2:
            central_box = tube[int(len(tube)/2)]
        else:
            central_box = tube[0]
        central_box = central_box[0:4]
        central_box = np.array([max(central_box[0], 0), max(central_box[1], 0), min(central_box[2], width - 1), min(central_box[3], height - 1)])
        central_box = np.insert(central_box[0:4], 0, id).reshape(1,-1)
        central_box = torch.from_numpy(central_box).float()
        return central_box

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

    return boxes, images, labels, num_tubes, paths#torch.stack(labels, dim=0)

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
