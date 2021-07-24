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
                       train=False):
        self.frames_per_tube = frames_per_tube
        self.min_frames_per_tube = min_frames_per_tube
        self.spatial_transform = spatial_transform
        self.make_function = make_function
        self.paths, self.labels, self.annotations = self.make_function()
        self.paths, self.labels, self.annotations = self.filter_data_without_tubelet()

        print('paths: {}, labels:{}, annot:{}'.format(len(self.paths), len(self.labels), len(self.annotations)))
        self.sampler = TubeCrop(tube_len=frames_per_tube, 
                                min_tube_len=min_frames_per_tube, 
                                central_frame=True,
                                max_num_tubes=max_num_tubes,
                                train=train)
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

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        annotation = self.annotations[index]
        boxes, segments, idxs = self.sampler(JSON_2_tube(annotation), annotation)

        if boxes == None or len(boxes) == 0:
            return None, None, None, None, None
        # if len(boxes)==0:
        #     return None, None, None, None
        # print('Loaded Boxes:', len(boxes))
        # print('segments: ', segments, len(segments))
        frames_names = []
        video_images = []
        num_tubes = len(segments)
        for seg in segments:
            # frames = list(itemgetter(*seg)(frames_paths))
            frames = [os.path.join(path,'frame{}.jpg'.format(i+1)) for i in seg] #rwf
            # frames = [os.path.join(path,'frame{:03}.jpg'.format(i+1)) for i in seg]
            frames_names.append(frames)
            tube_images = [] #one tube-16 frames
            for i in frames:
                img = self.spatial_transform(imread(i)) if self.spatial_transform else imread(i)
                tube_images.append(img)
            video_images.append(torch.stack(tube_images, dim=0))
        # if len(boxes) > self.max_num_tubes:
        #     idxs = random.sample(range(len(boxes)), self.max_num_tubes)
        #     boxes = list(itemgetter(*idxs)(boxes))
        #     video_images = list(itemgetter(*idxs)(video_images))
        
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
    def __init__(self, tube_len=16, min_tube_len=8, central_frame=True, max_num_tubes=4, train=True):
        """
        Args:
        """
        self.tube_len = tube_len
        self.min_tube_len = min_tube_len
        self.central_frame = central_frame
        self.max_num_tubes = max_num_tubes
        self.train = train

    def __call__(self, tubes: list, tube_path: str):
        # assert len(tubes) >= 1, "No tubes in video!!!==>{}".format(tube_path)
        # if len(tubes) < 1:
        #     return None, None, None
        
        segments = []
        boxes = []
        for tube in tubes:
            frames_idxs = self.__centered_frames__(tube['foundAt'])
            if len(frames_idxs) > 0:
                bbox = self.__central_bbox__(tube['boxes'], tube['id']+1)
                boxes.append(bbox)
                segments.append(frames_idxs)
        

        idxs = range(len(boxes))
        if self.max_num_tubes != 0 and len(boxes) > self.max_num_tubes:
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
        if len(tube_frames_idxs) < self.tube_len:
            last_idx = tube_frames_idxs[len(tube_frames_idxs)-1]
            tube_frames_idxs += (self.tube_len - len(tube_frames_idxs))*[last_idx]
            return tube_frames_idxs

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

    make_dataset = MakeRWF2000(root='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames', 
                                train=True,
                                path_annotations='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/Tubes/RWF-2000')
    dataset = TubeDataset(frames_per_tube=16, 
                            min_frames_per_tube=8, 
                            make_function=make_dataset,
                            spatial_transform=transforms.Compose([
                                transforms.CenterCrop(224),
                                transforms.ToTensor()
                            ]))
    # path, label, annotation,frames_names, boxes, video_images = dataset[213]
    # print('path: ', path)
    # print('label: ', label)
    # print('annotation: ', annotation)
    # print('boxes: ', boxes.size(), boxes)
    # print('video_images: ', video_images.size())
    # print('frames_names: ', frames_names)

    # videos_no_tubes = check_no_tubes(make_dataset)
    # print('videos_no_tubes: ', videos_no_tubes, len(videos_no_tubes))

    # def my_collate(batch):
    #     print('batch: ', len(batch), len(batch[0]), len(batch[1]), len(batch[2]), len(batch[3]), batch[3][0].size(), batch[3][1].size())
    #     # batch = filter(lambda img: img[0] is not None, batch)
    #     boxes = filter(lambda img: img[0] is not None, batch)
    #     video_images = filter(lambda img: img[1] is not None, batch)
    #     return data.dataloader.default_collate(list(boxes)), data.dataloader.default_collate(list(video_images)) 
        # return [list(boxes), list(video_images)]
    

    loader = DataLoader(dataset,
                        batch_size=4,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True,
                        collate_fn=my_collate)

    for i, data in enumerate(loader):
        # path, label, annotation,frames_names, boxes, video_images = data
        boxes, video_images, labels = data
        print('_____ {} ______'.format(i+1))
        # print('path: ', path)
        # print('label: ', label)
        # print('annotation: ', annotation)
        print('boxes: ', type(boxes), len(boxes), '-boxes[0]: ', boxes[0].size())
        print('video_images: ', type(video_images), len(video_images), '-video_images[0]: ', video_images[0].size())
        print('labels: ', type(labels), len(labels), '-labels: ', labels)
