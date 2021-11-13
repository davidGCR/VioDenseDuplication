

from torch.utils.data.sampler import WeightedRandomSampler
# 
# import src.VioNet.add_path
import imports
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
from customdatasets.make_dataset import *
from customdatasets.make_UCFCrime import *
from customdatasets.tube_crop import TubeCrop
from transformations.dynamic_image_transformation import DynamicImage
from global_var import *
from utils import natural_sort
from torch.utils.data.dataloader import default_collate
import json

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
                       config=None,
                       tube_sample_strategy=MIDDLE,
                       shape=(224,224)):
        self.config = config
        self.dataset = dataset
        self.shape = shape
        # self.input_type = input_type
        self.frames_per_tube = frames_per_tube
        self.tube_sample_strategy = tube_sample_strategy
        self.make_function = make_function
        if dataset == 'UCFCrime':
            self.paths, self.labels, _, self.annotations, self.num_frames = self.make_function()
            # indices_2_remove = []
            # for index in range(len(self.paths)):
            #     annotation = self.annotations[index]
            #     if len(annotation) == 0:
            #         indices_2_remove.append(index)
            # self.paths = [self.paths[i] for i in range(len(self.paths)) if i not in indices_2_remove]
            # self.labels = [self.labels[i] for i in range(len(self.labels)) if i not in indices_2_remove]
            # self.annotations = [self.annotations[i] for i in range(len(self.annotations)) if i not in indices_2_remove]
        elif dataset == 'UCFCrime_Reduced':
            self.paths, self.labels, self.annotations, self.num_frames = self.make_function()
        else:
            self.paths, self.labels, self.annotations = self.make_function()
            # self.paths, self.labels, self.annotations = filter_data_without_tubelet(self.paths, self.labels, self.annotations)

        # self.max_video_len = 39 if dataset=='hockey' else 149
        # self.keyframe = keyframe
        # self.spatial_transform_2 = spatial_transform_2

        print('paths: {}, labels:{}, annot:{}'.format(len(self.paths), len(self.labels), len(self.annotations)))
        self.sampler = TubeCrop(tube_len=frames_per_tube,
                                central_frame=True,
                                max_num_tubes=max_num_tubes,
                                train=train,
                                input_type=self.config['input_1']['type'],
                                sample_strategy=tube_sample_strategy,
                                random=random,
                                box_as_tensor=False)
        self.box_as_tensor = False
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
    
    def build_frame_name(self, path, frame_number, frames_names_list):
        if self.dataset == 'rwf-2000':
            return os.path.join(path,'frame{}.jpg'.format(frame_number+1))
        elif self.dataset == 'hockey':
            return os.path.join(path,'frame{:03}.jpg'.format(frame_number+1))
        elif self.dataset == 'RealLifeViolenceDataset':
            return os.path.join(path,'{:06}.jpg'.format(frame_number))
        elif self.dataset == 'UCFCrime':
            return os.path.join(path,'{:06}.jpg'.format(frame_number))
        elif self.dataset == 'UCFCrime_Reduced':
            frame_idx = frame_number
            pth = os.path.join(path, frames_names_list[frame_idx])
            return pth
    
    def __format_bbox__(self, bbox):
        """
        Format a tube bbox: [x1,y1,x2,y2] to a correct format
        """
        (width, height) = self.shape
        bbox = bbox[0:4]
        bbox = np.array([max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], width - 1), min(bbox[3], height - 1)])
        # bbox = np.insert(bbox[0:4], 0, id).reshape(1,-1).astype(float)
        bbox = bbox.reshape(1,-1).astype(float)
        if self.box_as_tensor:
            bbox = torch.from_numpy(bbox).float()
        return bbox
    
    def load_input_1(self, path, frames_indices, frames_names_list, sampled_tube):
        # print('\nload_input_1--> frames_paths')
        tube_images = []
        raw_clip_images = []
        tube_images_t = None
        tube_boxes = []
        if self.config['input_1']['type']=='rgb':
            frames_paths = [self.build_frame_name(path, i, frames_names_list) for i in frames_indices]
            # print('frames_paths: ', frames_paths)
            for i in frames_paths:
                img = imread(i)
                tube_images.append(img)
                _, frame_name = os.path.split(i)
                
                try:
                    box_idx = sampled_tube['frames_name'].index(frame_name)
                except Exception as e:
                    print("\nOops!", e.__class__, "occurred.")
                    print("sampled_tube['frames_name']: {}, frame: {} , sampled_indices: {}, path: {}".format(sampled_tube['frames_name'], frame_name, frames_indices, path))
                    exit()
                tube_boxes.append(box_idx)
            
            tube_boxes = [sampled_tube['boxes'][b] for b in tube_boxes]
            tube_boxes = [self.__format_bbox__(t) for t in tube_boxes]
            
            # print('\tube_boxes: ', tube_boxes, len(tube_boxes))
            raw_clip_images = tube_images.copy()
            if self.config['input_1']['spatial_transform']:
                tube_images_t, tube_boxes_t = self.config['input_1']['spatial_transform'](tube_images, tube_boxes)
       
       # elif self.config['input_1']['type']=='dynamic-image':
        #     tt = DynamicImage()
        #     for shot in seg:
        #         frames_paths = [self.build_frame_name(path, i, frames_names_list) for i in shot]
        #         shot_images = [imread(img_path) for img_path in frames_paths]
        #         img = self.spatial_transform(tt(shot_images)) if self.spatial_transform else tt(shot_images)
        #         tube_images.append(img)
        return tube_images_t, tube_boxes_t, tube_boxes, raw_clip_images
    
    def load_input_2(self, frames, path, frames_names_list):
        if self.config['input_2']['type'] == 'rgb':
            i = frames[int(len(frames)/2)]
            # print('central frame:', i)
            img_path = self.build_frame_name(path, i, frames_names_list)
            key_frame = imread(img_path)
            
        elif self.config['input_2']['type'] == 'dynamic-image':
            tt = DynamicImage()
            frames_paths = [self.build_frame_name(path, i, frames_names_list) for i in frames] #rwf
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

    def load_tube_from_file(self, annotation):
        if self.dataset == 'UCFCrime':
            return annotation
        else:
            if isinstance(annotation, list):
                video_tubes = annotation
            else:
                video_tubes = JSON_2_tube(annotation)
            assert len(video_tubes) >= 1, "No tubes in video!!!==>{}".format(annotation)
            return video_tubes
    
    def video_max_len(self, idx):
        path = self.paths[idx]
        if self.dataset == 'RealLifeViolenceDataset':
            max_video_len = len(os.listdir(path)) - 1
        elif self.dataset=='hockey':
            max_video_len = 39
        elif self.dataset=='rwf-2000':
            max_video_len = 149
        elif self.dataset == 'UCFCrime':
            max_video_len = self.annotations[idx][0]['foundAt'][-1]- 1
        elif self.dataset == 'UCFCrime_Reduced':
            max_video_len = len(os.listdir(path)) - 1
        return max_video_len

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        frames_names_list = os.listdir(path)
        frames_names_list = natural_sort(frames_names_list)
        # print('frames_names_list: ', frames_names_list)
        
        label = self.labels[index]
        annotation = self.annotations[index]
        
        # max_video_len = self.video_max_len(index)
        tubes_ = self.load_tube_from_file(annotation)

        assert len(tubes_)>0, "No tubes found: {}".format(path)

        #remove tubes with len=1
        # tubes_ = [t for t in tubes_ if t['len'] > 1]
        # print('\n\ntubes_: ', tubes_)
        sampled_frames_indices, chosed_tubes = self.sampler(tubes_)

        # for i in range(len(sampled_frames_indices)):
        #     print('\ntube[{}] \n (1)frames_names_list: {}, \n(2)tube frames_name: {}, \n(3)sampled_frames_indices: {}'.format(i,frames_names_list, chosed_tubes[i]['frames_name'], sampled_frames_indices[i]))
        # print('sampled_frames_indices: ', sampled_frames_indices)
        # print('boxes_from_sampler: ', boxes, boxes[0].shape)
        video_images = []
        final_tube_boxes = []
        num_tubes = len(sampled_frames_indices)
        for frames_indices, sampled_tube in zip(sampled_frames_indices, chosed_tubes):
            # print('\nload_input_1 args: ', path, frames_indices, boxes)
            # dup_boxes = boxes[0][:,1:5]
            tube_images_t, tube_boxes_t, tube_boxes, _ = self.load_input_1(path, frames_indices, frames_names_list, sampled_tube)
            video_images.append(torch.stack(tube_images_t, dim=0))
            m = int(len(tube_boxes)/2) #middle box from tube
            

            ##setting id to box
            c_box = tube_boxes_t[m]
            id_tensor = torch.tensor([0]).unsqueeze(dim=0).float()
            print('\n', ' id_tensor: ', id_tensor,id_tensor.size())
            print(' c_box: ', c_box, c_box.size(), ' index: ', m)

            if c_box.size(0)==0:
                print(' Here error: ', path, index, '\n',
                        c_box, '\n', 
                        sampled_tube, '\n', 
                        frames_indices, '\n', 
                        tube_boxes_t, len(tube_boxes_t), '\n', 
                        tube_boxes, len(tube_boxes))
            f_box = torch.cat([id_tensor , c_box], dim=1).float()
            
            final_tube_boxes.append(f_box)

        # #add extra dimension to boxes for id
        # final_tube_boxes = [None]*len(final_tube_boxes_tmp)
        # for i in range(len(final_tube_boxes_tmp)):
        #     id_tensor = torch.tensor([0]).unsqueeze(dim=0).float()
        #     print('\n', i, ' id_tensor: ', id_tensor,id_tensor.size())
        #     print(i, ' final_tube_boxes_tmp[i]: ', final_tube_boxes_tmp[i], final_tube_boxes_tmp[i].size())
        #     if final_tube_boxes_tmp[i].size(0)==0:
        #         print(i, ' Here error: ', final_tube_boxes_tmp[i], '\n', chosed_tubes[i])
        #     box = torch.cat([id_tensor , final_tube_boxes_tmp[i]], dim=1).float()
            
        #     # print('box: ', box, box.size())
        #     final_tube_boxes[i] = box
        
        key_frames = []
        if self.config['input_2'] is not None:
            for seg in sampled_frames_indices:
                key_frame, _ = self.load_input_2(seg, path, frames_names_list)
                key_frames.append(key_frame)
        
        #padding
        if len(video_images)<self.max_num_tubes:
            for i in range(self.max_num_tubes-len(video_images)):
                video_images.append(video_images[len(video_images)-1])
                p_box = tube_boxes[len(tube_boxes)-1]
                tube_boxes.append(p_box)
                if self.config['input_2'] is not None:
                    key_frames.append(key_frames[-1])

        final_tube_boxes = torch.stack(final_tube_boxes, dim=0).squeeze()
        
        if len(final_tube_boxes.shape)==1:
            final_tube_boxes = torch.unsqueeze(final_tube_boxes, dim=0)
            # print('boxes unsqueeze: ', boxes)
        
        video_images = torch.stack(video_images, dim=0)#.permute(0,4,1,2,3)#.permute(0,2,1,3,4)
        if self.config['input_2'] is not None:
            key_frames = torch.stack(key_frames, dim=0)
            if torch.isnan(key_frames).any().item():
                print('Detected Nan at: ', path)
            if torch.isinf(key_frames).any().item():
                print('Detected Inf at: ', path)
            # print('video_images: ', video_images.size())
            # print('key_frames: ', key_frames.size())
            # print('final_tube_boxes: ', final_tube_boxes,  final_tube_boxes.size())
            # print('label: ', label)
            return final_tube_boxes, video_images, label, num_tubes, path, key_frames
        else:
            return final_tube_boxes, video_images, label, num_tubes, path

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
        # print('--->', i, boxes[i])

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

from transformations.data_aug.data_aug import *
from transformations.vizualize_batch import *
from model_transformations import i3d_video_transf, resnet_transf

from torch.utils.data import DataLoader
from torchvision import transforms

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
    #                         spatial_transform=None)
    
    ann_file  = ('Train_annotation.pkl', 'Train_normal_annotation.pkl')# if train else ('Test_annotation.pkl', 'Test_normal_annotation.pkl')
    home_path = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local'
    make_dataset = MakeUCFCrime(
            root=os.path.join(home_path, 'UCFCrime_Reduced', 'frames'), 
            sp_abnormal_annotations_file=os.path.join(home_path,'VioNetDB-splits/UCFCrime', ann_file[0]), 
            sp_normal_annotations_file=os.path.join(home_path,'VioNetDB-splits/UCFCrime', ann_file[1]), 
            action_tubes_path=os.path.join(home_path,'ActionTubesV2/UCFCrime_Reduced'),
            train=True,
            ground_truth_tubes=False)
    
    TWO_STREAM_INPUT_train = {
        'input_1': {
            'type': 'rgb',
            # 'spatial_transform': i3d_video_transf()['train'],
            'spatial_transform': Compose(
                [
                    # ClipRandomHorizontalFlip(), 
                    ClipRandomScale(scale=0.2, diff=True), 
                    # ClipRandomRotate(angle=5),
                    ClipRandomTranslate(translate=0.1, diff=True),
                    NumpyToTensor()
                ],
                probs=[1, 1]
                ),
            'temporal_transform': None
        },
        'input_2': {
            'type': 'rgb',
            'spatial_transform': resnet_transf()['train'],
            'temporal_transform': None
        }
        # 'input_2': {
        #     'type': 'dynamic-image',
        #     'spatial_transform': resnet_di_transf()['train'],
        #     'temporal_transform': None
        # }
    }
    train_dataset = TubeDataset(frames_per_tube=16, 
                            make_function=make_dataset,
                            max_num_tubes=1,
                            train=True,
                            dataset='UCFCrime_Reduced',
                            random=True,
                            config=TWO_STREAM_INPUT_train)
    
    # for i in range(len(train_dataset)):
    #     data = train_dataset[i]
    #     bboxes, video_images, label, num_tubes, path, key_frames = data
    #     if os.path.split(path)[1]=='Assault027_x264':
    #         print(i)
    #         break
    random.seed(34)
    for i in range(10):
        bboxes, video_images, label, num_tubes, path, key_frames = train_dataset[204]
        print('\tpath: ', path)
        print('\tvideo_images: ', type(video_images), video_images.size())
        print('\tbboxes: ', bboxes.size())
    

    # frames_numpy = video_images.cpu().numpy()
    # # bboxes_numpy = torch.unsqueeze(bboxes, dim=0).cpu().numpy()
    # bboxes_numpy = np.array([bboxes.cpu().numpy()[:,1:5]]*16).reshape((1,16,4))
    # print('\tframes_numpy: ', frames_numpy.shape)
    # print('\tbboxes_numpy: ', bboxes_numpy, bboxes_numpy.shape)
    # for j in range(frames_numpy.shape[0]):
    #     plot_clip(frames_numpy[j], bboxes_numpy[j], (4,4))

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

    # crop = TubeCrop()
    # lp = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    # print('lp len:', len(lp))
    # idxs = crop.__centered_frames__(lp)
    # print('idxs: ', idxs, len(idxs))
