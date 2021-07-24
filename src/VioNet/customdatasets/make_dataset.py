import os
import glob
from operator import itemgetter
import numpy as np
import json
import re

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

        

class MakeRWF2000():
    def __init__(self, root, train, path_annotations=None, path_feat_annotations=None):
        self.root = root
        self.train = train
        self.path_annotations = path_annotations
        self.path_feat_annotations = path_feat_annotations
        # self.F_TAG = "Fight"
        # self.NF_TAG = "NonFight"
        self.classes = ["NonFight", "Fight"]
    
    def classes(self):
        return self.classes
    
    def split(self):
        split = "train" if self.train else "val"
        return split
    
    def __call__(self):
        split = self.split()
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

class MakeUCFCrime2LocalClips():
    def __init__(self, root):
        self.root = root
        # self.annotation_path = annotation_path
        # self.bbox_path = bbox_path
    
    def __get_list__(self, path):
        paths = os.listdir(path)
        # labels = []
        paths = [os.path.join(path,pt) for pt in paths if os.path.isdir(os.path.join(path,pt))]

        # for path in paths:
        #     label = 0 if "Normal" in path else 1
        #     labels.append(label)
        
        return paths

    def __call__(self):
        if isinstance(self.root,tuple):
            abnormal_paths = self.__get_list__(self.root[0])
            normal_paths = self.__get_list__(self.root[1])
            normal_paths = [path for path in normal_paths if "Normal" in path]
            paths = abnormal_paths + normal_paths
            # paths = list(set(paths))
            
            labels = [1]*len(abnormal_paths) + [0]*len(normal_paths)
        else:
            print("No tuple!!!")
            # paths = self.__get_list__(self.root)
        
        # labels = []
        # for path in paths:
        #     label = 0 if "Normal" in path else 1
        #     labels.append(label)
        return paths, labels
        

from collections import Counter
import random

if __name__=="__main__":
    # make_func = MakeRWF2000(root='/media/david/datos/Violence DATA/RWF-2000/frames', 
    #                 train=True,
    #                 path_annotations='/media/david/datos/Violence DATA/Tubes/RWF-2000')
    # paths, labels, annotations = make_func()
    # print("paths: ", paths[0:10], len(paths))
    # print("labels: ", labels[0:10], len(labels))
    # print("annotations: ", annotations[0:10], len(annotations))

    make_func = MakeHockeyDataset(root='/media/david/datos/Violence DATA/DATASETS/HockeyFightsDATASET/frames', 
                    train=False,
                    cv_split_annotation_path='/media/david/datos/Violence DATA/VioDB/hockey_jpg1.json',
                    path_annotations='/media/david/datos/Violence DATA/ActionTubes/hockey')
    paths, labels, annotations = make_func()
    print("paths: ", paths[0:10], len(paths))
    print("labels: ", labels[0:10], len(labels))
    print("annotations: ", annotations[0:10], len(annotations))

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

    # m = MakeUCFCrime2LocalClips(root=('/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips',
    #                              '/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/frames'))
    # paths, labels = m()
    # idx= random.randint(0, len(paths)-1)
    # print(Counter(labels))
    # print(paths[idx])
    # print(labels[idx])