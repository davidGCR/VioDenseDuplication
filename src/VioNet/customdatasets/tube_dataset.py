import torch.utils.data as data
import numpy as np
import torch
from torch.utils.data import dataset
from operator import itemgetter
import os
import random
from operator import itemgetter
from customdatasets.dataset_utils import imread
from customdatasets.make_dataset import MakeRWF2000

class TubeDataset(data.Dataset):
    """
    Load videos stored as images folders.
    Use this to load raw frames from tubes.
    """

    def __init__(self, frames_per_tube,
                       min_frames_per_tube,
                       make_function,
                       spatial_transform=None,
                       return_metadata=False,
                       max_num_tubes=16):
        self.frames_per_tube = frames_per_tube
        self.min_frames_per_tube = min_frames_per_tube
        self.spatial_transform = spatial_transform
        self.make_function = make_function
        self.paths, self.labels, self.annotations = self.make_function()
        self.sampler = TubeCrop(tube_len=frames_per_tube, min_tube_len=min_frames_per_tube, central_frame=True)
        self.return_metadata = return_metadata
        self.max_num_tubes = max_num_tubes
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        annotation = self.annotations[index]
        boxes, segments = self.sampler(JSON_2_tube(annotation), annotation)

        if boxes == None:
            return None, None, None
        # print('Loaded Boxes:', len(boxes))
        # print('segments: ', segments, len(segments))
        frames_names = []
        video_images = []
        for seg in segments:
            # frames = list(itemgetter(*seg)(frames_paths))
            frames = [os.path.join(path,'frame{}.jpg'.format(i+1)) for i in seg]
            frames_names.append(frames)
            tube_images = [] #one tube-16 frames
            for i in frames:
                img = self.spatial_transform(imread(i)) if self.spatial_transform else imread(i)
                tube_images.append(img)
            video_images.append(torch.stack(tube_images, dim=0))
        if len(boxes) > self.max_num_tubes:
            idxs = random.sample(range(len(boxes)), self.max_num_tubes)
            boxes = list(itemgetter(*idxs)(boxes))
            video_images = list(itemgetter(*idxs)(video_images))
        boxes = torch.stack(boxes, dim=0).squeeze()
        video_images = torch.stack(video_images, dim=0)
        # return path, label, annotation, frames_names, boxes, video_images
        return boxes, video_images, label



class TubeCrop(object):
    def __init__(self, tube_len=16, min_tube_len=8, central_frame=True):
        """
        Args:
        """
        self.tube_len = tube_len
        self.min_tube_len = min_tube_len
        self.central_frame = central_frame

    def __call__(self, tubes: list, tube_path: str):
        # assert len(tubes) >= 1, "No tubes in video!!!==>{}".format(tube_path)
        # if len(tubes) < 1:
        #     return None, None
        segments = []
        boxes = []
        for tube in tubes:
            frames_idxs = self.__centered_frames__(tube['foundAt'])
            if len(frames_idxs) > 0:
                bbox = self.__central_bbox__(tube['boxes'], tube['id']+1)
                boxes.append(bbox)
                segments.append(frames_idxs)
        
        if len(boxes) < 1:
            return None, None
        
        return boxes, segments
    
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
            return []

    def __central_bbox__(self, tube, id):
        if len(tube)>2:
            central_box = tube[int(len(tube)/2)]
        else:
            central_box = tube[0]

        central_box = np.insert(central_box[0:4], 0, id).reshape(1,-1)
        central_box = torch.from_numpy(central_box).float()
        return central_box

def my_collate(batch):
    # print('BATCH: ', type(batch), len(batch), len(batch[0]))
    boxes = [item[0] for item in batch if item[0] is not None]
    images = [item[1] for item in batch if item[1] is not None]
    labels = [item[2] for item in batch if item[2] is not None]
    # print('BATCH filtered: ', len(boxes), len(images), len(labels))
    # target = torch.LongTensor(target)

    # print('boxes:', type(boxes), len(boxes))
    # print('images:', type(boxes), len(images))
    # print('boxes[i]:', type(boxes[0]), boxes[0].size())
    # print('images[i]:', type(images[0]), images[0].size())
    # return [torch.stack(boxes, dim=0), torch.stack(images, dim=0)]
    return boxes, images, labels

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
