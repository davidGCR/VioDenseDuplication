import os
import glob
from operator import itemgetter
import numpy as np
import json
import re
import random

class MakeImageHMDB51():
    def __init__(self, root, annotation_path, fold, train):
        self.root = root
        self.annotation_path = annotation_path
        self.fold = fold
        self.train = train
        self.TRAIN_TAG = 1
        self.TEST_TAG = 2
    
    def __select_fold__(self, video_list):
        target_tag = self.TRAIN_TAG if self.train else self.TEST_TAG
        split_pattern_name = "*test_split{}.txt".format(self.fold)
        split_pattern_path = os.path.join(self.annotation_path, split_pattern_name)
        annotation_paths = glob.glob(split_pattern_path)
        # for a in annotation_paths:
        #     print(a)
        selected_files = []
        for filepath in annotation_paths:
            with open(filepath) as fid:
                lines = fid.readlines()
            for line in lines:
                video_filename, tag_string = line.split()
                tag = int(tag_string)
                if tag == target_tag:
                    selected_files.append(video_filename[:-4])
        selected_files = set(selected_files)

        # print(selected_files, len(selected_files))
        indices = []
        for video_index, video_path in enumerate(video_list):
            # print(os.path.basename(video_path))
            if os.path.basename(video_path) in selected_files:
                indices.append(video_index)

        return indices

    def __call__(self):
        # classes = sorted(os.listdir(self.root))
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        # print("classes:",classes)
        # print("class_to_idx:",class_to_idx)
        paths = []
        labels = []
        for c in classes:
            class_path = os.path.join(self.root, c)
            for v in os.scandir(class_path):
                if v.is_dir():
                    video_path = os.path.join(class_path,v.name)
                    paths.append(video_path)
                    labels.append(class_to_idx[c])

        # print(paths, len(paths))
        indices = self.__select_fold__(paths)
        paths = list(itemgetter(*indices)(paths))
        labels = list(itemgetter(*indices)(labels))
        # print(paths, len(paths))
        # print(labels, len(labels))
        return paths, labels

class MakeHockeyDataset():
    def __init__(self, root, train, cv_split_annotation_path, path_annotations=None):
        self.root = root
        self.train = train
        self.path_annotations = path_annotations
        self.cv_split_annotation_path = cv_split_annotation_path
        # self.split = split
        # self.F_TAG = "Fight"
        # self.NF_TAG = "NonFight"
        self.classes = ["nonviolence","violence"]
    
    def split(self):
        split = "training" if self.train else "validation"
        return split
    
    def load_annotation_data(self):
        with open(self.cv_split_annotation_path, 'r') as data_file:
            return json.load(data_file)
    
    def get_video_names_and_labels(self, data, split):
        video_names = []
        video_labels = []
        annotations = []

        for key, val in data['database'].items():
            if val['subset'] == split:
                label = val['annotations']['label']
                cl = 'violence' if label=='fi' else 'nonviolence'

                label = 0 if label=='no' else 1
                v_name = re.findall(r'\d+', key)[0]
                folder = os.path.join(self.root, cl, v_name)
                assert os.path.isdir(folder), "Folder:{} does not exist!!!".format(folder)
                video_names.append(folder)
                video_labels.append(label)
                if self.path_annotations:
                    ann_file = os.path.join(self.path_annotations, cl, v_name+'.json')
                    assert os.path.isfile(ann_file), "Annotation file:{} does not exist!!!".format(ann_file)
                    annotations.append(ann_file)

        return video_names, video_labels, annotations
    
    def __call__(self):
        data = self.load_annotation_data()
        split = self.split()
        paths, labels, annotations = self.get_video_names_and_labels(data, split)
        return paths, labels, annotations

class MakeRLVDDataset():
    def __init__(self, root, train, cv_split_annotation_path, path_annotations=None):
        self.root = root
        self.train = train
        self.path_annotations = path_annotations
        self.cv_split_annotation_path = cv_split_annotation_path
        # self.split = split
        # self.F_TAG = "Fight"
        # self.NF_TAG = "NonFight"
        self.classes = ["NonViolence","Violence"]
    
    def split(self):
        split = "training" if self.train else "validation"
        return split
    
    def load_annotation_data(self):
        with open(self.cv_split_annotation_path, 'r') as data_file:
            return json.load(data_file)
    
    def get_video_names_and_labels(self, data, split):
        video_names = []
        video_labels = []
        annotations = []
        # num_frames = []

        for key, val in data['database'].items():
            if val['subset'] == split:
                label = val['annotations']['label']
                cl = 'Violence' if label=='fi' else 'NonViolence'

                label = 0 if label=='no' else 1
                # v_name = re.findall(r'\d+', key)[0]
                v_name = key
                folder = os.path.join(self.root, cl, v_name)
                assert os.path.isdir(folder), "Folder:{} does not exist!!!".format(folder)
                video_names.append(folder)
                video_labels.append(label)
                n = os.listdir(folder)
                # n = [img for img in n if '.jpg' in img]
                # num_frames.append(len(n))
                if self.path_annotations:
                    ann_file = os.path.join(self.path_annotations, cl, v_name+'.json')
                    assert os.path.isfile(ann_file), "Annotation file:{} does not exist!!!".format(ann_file)
                    annotations.append(ann_file)

        return video_names, video_labels, annotations
    
    def __call__(self):
        data = self.load_annotation_data()
        split = self.split()
        paths, labels, annotations = self.get_video_names_and_labels(data, split)
        return paths, labels, annotations


        
CATEGORY_ALL = 2
CATEGORY_POS = 1
CATEGORY_NEG = 0

class MakeRWF2000():
    def __init__(self, 
                root,
                train,
                category=CATEGORY_ALL,
                path_annotations=None, 
                path_feat_annotations=None,
                shuffle=False):
        self.root = root
        self.train = train
        self.path_annotations = path_annotations
        self.path_feat_annotations = path_feat_annotations
        # self.F_TAG = "Fight"
        # self.NF_TAG = "NonFight"
        self.classes = ["NonFight", "Fight"]
        self.category = category
        self.shuffle = shuffle
    
    def classes(self):
        return self.classes
    
    def split(self):
        split = "train" if self.train else "val"
        return split
    
    def all_categories(self, split):
        paths = []
        labels = []
        annotations = []
        feat_annotations = []
        for idx, cl in enumerate(self.classes):
            for video_sample in os.scandir(os.path.join(self.root, split, cl)):
                if video_sample.is_dir():
                    paths.append(os.path.join(self.root, split, cl, video_sample))
                    labels.append(idx)
                    if self.path_annotations:
                        assert os.path.exists(os.path.join(self.path_annotations, split, cl, video_sample.name +'.json')), "Annotation does not exist!!!"
                        annotations.append(os.path.join(self.path_annotations, split, cl, video_sample.name +'.json'))
                    if self.path_feat_annotations:
                        assert os.path.exists(os.path.join(self.path_feat_annotations, split, cl, video_sample.name +'.txt')), "Feature annotation does not exist!!!"
                        feat_annotations.append(os.path.join(self.path_feat_annotations, split, cl, video_sample.name +'.txt'))
        
        return paths, labels, annotations
    
    def positive_category(self, split):
        paths = []
        labels = []
        annotations = []
        feat_annotations = []
        label = 1
        label_name = self.classes[label]
        for video_sample in os.scandir(os.path.join(self.root, split, label_name)):
            if video_sample.is_dir():
                paths.append(os.path.join(self.root, split, label_name, video_sample))
                labels.append(label)
                if self.path_annotations:
                    assert os.path.exists(os.path.join(self.path_annotations, split, label_name, video_sample.name +'.json')), "Annotation does not exist!!!"
                    annotations.append(os.path.join(self.path_annotations, split, label_name, video_sample.name +'.json'))
                if self.path_feat_annotations:
                    assert os.path.exists(os.path.join(self.path_feat_annotations, split, label_name, video_sample.name +'.txt')), "Feature annotation does not exist!!!"
                    feat_annotations.append(os.path.join(self.path_feat_annotations, split, label_name, video_sample.name +'.txt'))
        
        return paths, labels, annotations
    
    def negative_category(self, split):
        paths = []
        labels = []
        annotations = []
        feat_annotations = []
        label = 0
        label_name = self.classes[label]
        for video_sample in os.scandir(os.path.join(self.root, split, label_name)):
            if video_sample.is_dir():
                paths.append(os.path.join(self.root, split, label_name, video_sample))
                labels.append(label)
                if self.path_annotations:
                    assert os.path.exists(os.path.join(self.path_annotations, split, label_name, video_sample.name +'.json')), "Annotation does not exist!!!"
                    annotations.append(os.path.join(self.path_annotations, split, label_name, video_sample.name +'.json'))
                if self.path_feat_annotations:
                    assert os.path.exists(os.path.join(self.path_feat_annotations, split, label_name, video_sample.name +'.txt')), "Feature annotation does not exist!!!"
                    feat_annotations.append(os.path.join(self.path_feat_annotations, split, label_name, video_sample.name +'.txt'))
        
        return paths, labels, annotations
    
    def __call__(self):
        split = self.split()
        if self.category == CATEGORY_ALL:
            paths, labels, annotations =  self.all_categories(split)
        elif self.category == CATEGORY_POS:
            paths, labels, annotations = self.positive_category(split)
        elif self.category == CATEGORY_NEG:
            paths, labels, annotations = self.negative_category(split)
        
        if self.shuffle:
            c = list(zip(paths, labels, annotations))
            random.shuffle(c)
            paths, labels, annotations = zip(*c)
        
        return paths, labels, annotations
        

class MakeUCFCrime2Local():
    def __init__(self, root, annotation_path, bbox_path, train):
        self.root = root
        self.annotation_path = annotation_path
        self.bbox_path = bbox_path
        self.train = train
 
    def split(self):
        split = "Train_split_AD.txt" if self.train else "Test_split_AD.txt"
        return split
    
    def sp_annotation(self, path):
        """
        1   Track ID. All rows with the same ID belong to the same path.
        2   xmin. The top left x-coordinate of the bounding box.
        3   ymin. The top left y-coordinate of the bounding box.
        4   xmax. The bottom right x-coordinate of the bounding box.
        5   ymax. The bottom right y-coordinate of the bounding box.
        6   frame. The frame that this annotation represents.
        7   lost. If 1, the annotation is outside of the view screen.
        8   occluded. If 1, the annotation is occluded.
        9   generated. If 1, the annotation was automatically interpolated.
        10  label. The label for this annotation, enclosed in quotation marks.
        11+ attributes. Each column after this is an attribute.
        """
        assert os.path.isfile(path), "Txt Annotation {} Not Found!!!".format(path)

        annotations = []
        with open(path) as fid:
            lines = fid.readlines()
            ss = 1 if lines[0].split()[5] == '0' else 0
            for line in lines:
                # v_name = line.split()[0]
                # print(line.split())
                ann = line.split()
                frame_number = int(ann[5]) + ss
                valid = ann[6]
                if valid == '0':
                    annotations.append(
                        {
                            "frame": frame_number,
                            "xmin": ann[1],
                            "ymin": ann[2],
                            "xmax": ann[3],
                            "ymax": ann[4]
                        }
                    )
        positive_intervals = self.positive_segments(annotations)
        return annotations, positive_intervals
                    
    def positive_segments(self, annotations):
        frames = []
        positive_intervals = []
        for an in annotations:
            frames.append(int(an["frame"]))
        frames.sort()
        start_end = np.diff((np.diff(frames) == 1) + 0, prepend=0, append=0)
        # Look for where it flips from 1 to 0, or 0 to 1.
        start_idx = np.where(start_end == 1)[0]
        end_idx = np.where(start_end == -1)[0]

        # print("---- start_idx", start_idx)
        # print("---- end_idx", end_idx, end_idx.shape)
        for s, e in zip(start_idx,end_idx):
            # print("---- ", s,e)
            # print("[{},{}]".format(frames[s], frames[e]))
            positive_intervals.append((frames[s], frames[e]))
        
        return positive_intervals

    def __call__(self):
        split_file = os.path.join(self.annotation_path, self.split())
        paths = []
        labels = []
        annotations = []
        positive_intervals = []
        with open(split_file) as fid:
            lines = fid.readlines()
            for line in lines:
                v_name = line.split()[0]
                # print(v_name[0])
                if os.path.isdir(os.path.join(self.root, v_name)):
                    paths.append(os.path.join(self.root, v_name))
                    label = 0 if "Normal" in v_name else 1
                    labels.append(label)
                    if label==1:
                        annotation, intervals = self.sp_annotation(os.path.join(self.bbox_path, v_name+".txt"))
                        annotations.append(annotation)
                        positive_intervals.append(intervals)
                    else:
                        annotations.append(None)
                        positive_intervals.append(None)
                else:
                    print("Folder ({}) not found!!!".format(v_name))
        
        return paths, labels, annotations, positive_intervals

import cv2


class MakeUCFCrime2LocalClips():
    def __init__(self, root, path_annotations, path_person_detections, abnormal):
        self.root = root
        self.path_annotations = path_annotations
        self.path_person_detections = path_person_detections
        self.classes = ['normal', 'anomaly'] #Robbery,Stealing
        self.subclasses = ['Arrest', 'Assault'] #Robbery,Stealing
        self.abnormal = abnormal
    
    def __get_list__(self, path):
        paths = os.listdir(path)
        paths = [os.path.join(path,pt) for pt in paths if os.path.isdir(os.path.join(path,pt))]
        return paths
    
    def __annotation__(self, folder_path):
        v_name = os.path.split(folder_path)[1]
        annotation = [ann_file for ann_file in os.listdir(self.path_annotations) if ann_file.split('.')[0] in v_name.split('(')]
        annotation = annotation[0]
        # print('annotation: ',annotation)
        return os.path.join(self.path_annotations, annotation)

    # def __annotation_p_detections__(self, folder_path):

    
    def ground_truth_boxes(self, video_folder, ann_path):
        frames = os.listdir(video_folder)
        frames_numbers = [int(re.findall(r'\d+', f)[0]) for f in frames]
        frames_numbers.sort()
        # print(frames_numbers)

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
        # tmp = [a['frame'] for a in annotations]
        # print(tmp)
        
        return annotations
    
    def plot(self, folder_imgs, annotations_dict, live_paths=[]):
        imgs = os.listdir(folder_imgs)
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            '''
            alist.sort(key=natural_keys) sorts in human order
            http://nedbatchelder.com/blog/200712/human_sorting.html
            (See Toothy's implementation in the comments)
            '''
            return [ atoi(c) for c in re.split(r'(\d+)', text) ]
        
        imgs.sort(key=natural_keys)
        # print(type(folder_imgs),type(f_paths[0]))
        f_paths = [os.path.join(folder_imgs, ff) for ff in imgs]
        
        for img_path in f_paths:
            print(img_path)
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            f_num = os.path.split(img_path)[1]
            f_num = int(re.findall(r'\d+', f_num)[0])
            ann = [ann for ann in annotations_dict if ann['frame']==f_num][0]
            x1 = ann["xmin"]
            y1 = ann["ymin"]
            x2 = ann["xmax"]
            y2 = ann["ymax"]
            cv2.rectangle(image,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0,238,238),
                            1)
            if len(live_paths)>0:
                frame = img_path.split('/')[-1]
                
                for l in range(len(live_paths)):
                    
                    foundAt = True if frame in live_paths[l]['frames_name'] else False
                    if foundAt:
                        idx = live_paths[l]['frames_name'].index(frame)
                        bbox = live_paths[l]['boxes'][idx]
                        x1 = bbox[0]
                        y1 = bbox[1]
                        x2 = bbox[2]
                        y2 = bbox[3]
                        cv2.rectangle(image,
                                    (int(x1), int(y1)),
                                    (int(x2), int(y2)),
                                    (255,0,0),
                                    1)
            cv2.namedWindow('FRAME'+str(f_num),cv2.WINDOW_NORMAL)
            cv2.resizeWindow('FRAME'+str(f_num), (600,600))
            image = cv2.resize(image, (600,600))
            cv2.imshow('FRAME'+str(f_num), image)
            key = cv2.waitKey(250)#pauses for 3 seconds before fetching next image
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows()

    def __call__(self):
        root_anomaly = os.path.join(self.root, self.classes[1])
        root_normal = os.path.join(self.root, self.classes[0])

        # root_anomaly_p_detections = os.path.join(self.path_person_detections, self.classes[1])
        
        if self.abnormal:
            abnormal_paths = self.__get_list__(root_anomaly)
            paths = abnormal_paths
            annotations_anomaly = [self.__annotation__(pt) for pt in abnormal_paths]
            annotations = annotations_anomaly
            labels = [1]*len(abnormal_paths)
            annotations_p_detections = []
            num_frames = []
            for ap in abnormal_paths:
                assert os.path.isdir(ap), 'Folder does not exist!!!'
                n = len(os.listdir(ap))
                num_frames.append(n)
                sp = ap.split('/')
                p_path = os.path.join(self.path_person_detections, sp[-2], sp[-1]+'.json')
                assert os.path.isfile(p_path), 'P_annotation does not exist!!!'
                annotations_p_detections.append(p_path)

        else:
            normal_paths = self.__get_list__(root_normal)
            normal_paths = [path for path in normal_paths if "Normal" in path]
            paths = normal_paths
            annotations_normal = [None]*len(normal_paths)
            annotations = annotations_normal
            labels = [0]*len(normal_paths)
            annotations_p_detections = [None]*len(normal_paths)
            num_frames = []
            for ap in normal_paths:
                assert os.path.isdir(ap), 'Folder does not exist!!!'
                n = len(os.listdir(ap))
                num_frames.append(n)
        # paths = abnormal_paths + normal_paths
        # annotations = annotations_anomaly + annotations_normal
        # labels = [1]*len(abnormal_paths) + [0]*len(normal_paths)
        
        return paths, labels, annotations, annotations_p_detections, num_frames
        

from collections import Counter
import random
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

def _avg_num_tubes(annotations):
    video_num_tubes=[]
    num_tubes=[]
    tube_lengths = []
    for ann in annotations:
        tubes = JSON_2_tube(ann)
        video_num_tubes.append((ann, len(tubes)))
        num_tubes.append(len(tubes))
        for tube in tubes:
            # print('tube[len]:', tube['len'], len(tube['boxes']), len(tube['foundAt']))
            l = 16 if tube['len']>16 else tube['len']
            tube_lengths.append(tube['len'])
    
    def Average(lst):
        return sum(lst) / len(lst)
    
    print('Avg num_tubes: ', Average(num_tubes))
    print('Avg len_tubes: ', Average(tube_lengths))

def _get_num_tubes(annotations, make_func):
    video_num_tubes=[]
    num_tubes=[]
    for ann in annotations:
        tubes = JSON_2_tube(ann)
        video_num_tubes.append((ann, len(tubes)))
        num_tubes.append(len(tubes))
    with open('hockey_num_tubes_{}.txt'.format('train' if make_func.train else 'val'), 'w') as filehandle:
        filehandle.writelines("{},{}\n".format(t[0], t[1]) for t in video_num_tubes)
    
   
    
if __name__=="__main__":
    make_func = MakeRWF2000(root='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames', 
                    train=True,
                    path_annotations='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/ActionTubes/final/rwf',
                    category=2)
    paths, labels, annotations = make_func()
    print("paths: ", len(paths))
    print("labels: ",len(labels))
    print("annotations: ",len(annotations))

    _avg_num_tubes(annotations)

    # print("no tubes in: ")
    # without_tube=[]
    # for ann in annotations:
    #     tubes = JSON_2_tube(ann)
    #     if len(tubes)==0:
    #         # print(len(tubes))
    #         without_tube.append(ann)
    
    # with open('3without_tube_{}.txt'.format('train' if make_func.train else 'val'), 'w') as filehandle:
    #     filehandle.writelines("%s\n" % t for t in without_tube)

    # tubes = JSON_2_tube('/media/david/datos/Violence DATA/ActionTubes/RWF-2000/train/Fight/C8wt47cphU8_0.json')
    # print("tubes: ",len(tubes))

    
    ###################################################################################################################################
    # make_func = MakeHockeyDataset(root='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/HockeyFightsDATASET/frames', 
    #                 train=False,
    #                 cv_split_annotation_path='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/VioNetDB-splits/hockey_jpg1.json',
    #                 path_annotations='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/ActionTubes/final/hockey')
    # paths, labels, annotations = make_func()
    # print("paths: ", len(paths))
    # print("labels: ", len(labels))
    # print("annotations: ", len(annotations))

    # _avg_num_tubes(annotations)
    # _get_num_tubes(annotations, make_func)
    ###################################################################################################################################

    # m = MakeUCFCrime2Local(root='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/frames',
    #                         annotation_path='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/readme',
    #                         bbox_path='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/readme/Txt annotations',
    #                         train=False)
    # paths, labels, annotations, intervals = m()
    # idx=22
    # print(paths[idx])
    # print(labels[idx])
    # print(annotations[idx][0:10])
    # print(intervals[idx])

    # m = MakeUCFCrime2LocalClips(root_anomaly='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips/anomaly',
    #                             root_normal='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/frames',
    #                             path_annotations='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CrimeViolence2LocalDATASET/Txt annotations-longVideos')
    # paths, labels, annotations = m()
    # # idx= random.randint(0, len(paths)-1)
    # idx=65
    # print(idx)
    # print(Counter(labels))
    # print(paths[idx])
    # print(labels[idx])
    # print(annotations[idx])

    # anns = m.ground_truth_boxes(paths[idx],annotations[idx])
    # m.plot(paths[idx], anns)

    ###################################################################################################################################
    # make_func = MakeRLVDDataset(root='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RealLifeViolenceDataset/frames', 
    #                 train=False,
    #                 cv_split_annotation_path='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/VioNetDB-splits/RealLifeViolenceDataset1.json',
    #                 path_annotations='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/ActionTubes/RealLifeViolenceDataset')
    # paths, labels, annotations, num_frames = make_func()
    # print("paths: ", len(paths))
    # print("labels: ", len(labels))
    # print("annotations: ", len(annotations))
    # print("num_frames: ", len(num_frames))
    # _avg_num_tubes(annotations)

    # print(paths[33:40])
    # print(labels[33:40])
    # print(annotations[33:40])
    # print(num_frames[33:40])
