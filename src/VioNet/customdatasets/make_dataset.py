import os
import glob
from operator import itemgetter

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

class MakeRWF2000():
    def __init__(self, root, train):
        self.root = root
        self.train = train
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
        for idx, cl in enumerate(self.classes):
            for video_sample in os.scandir(os.path.join(self.root, split, cl)):
                if video_sample.is_dir():
                    paths.append(os.path.join(self.root, split, cl, video_sample))
                    labels.append(idx)
        
        return paths, labels

if __name__=="__main__":
    m = MakeRWF2000(root="/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames", train=True)
    paths, labels = m()
    print("paths: ", paths, len(paths))
    print("labels: ", labels, len(labels))