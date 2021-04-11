import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import random


def read_features(file_path, features_dim=4096, max_features=32):
	if not os.path.exists(file_path):
		raise Exception(f"Feature doesn't exist: {file_path}")
	features = None
	with open(file_path, 'r') as fp:
		data = fp.read().splitlines(keepends=False)
		features = np.zeros((len(data), features_dim))
		for i, line in enumerate(data):
			features[i, :] = [float(x) for x in line.split(' ')]
	features = features[0:max_features, :]
	return torch.from_numpy(features).float()

class FeaturesLoader(Dataset):
    def __init__(self,
                 features_path,
                 annotation_path,
                 bucket_size=30,
                 features_dim=4096):

        super(FeaturesLoader, self).__init__()
        self.features_path = features_path
        self.bucket_size = bucket_size
        self.features_dim = features_dim
        # load video list
        self.state = 'Normal'
        self.features_list_normal, self.features_list_anomaly = FeaturesLoader._get_features_list(
            features_path=self.features_path,
            annotation_path=annotation_path)

        # print("features_list_anomaly:",self.features_list_anomaly)
        # print("features_list_normal:",self.features_list_normal)
        self.normal_i, self.anomalous_i = 0, 0
        

        self.shuffle()

    def shuffle(self):
        self.features_list_anomaly = np.random.permutation(self.features_list_anomaly)
        self.features_list_normal = np.random.permutation(self.features_list_normal)

    def __len__(self):
        return self.bucket_size * 2
    
    def __padding__(self, features):
        idx = random.randint(0, features.size()[0] - 1)
        lf = features[idx]
        lf = torch.stack((self.bucket_size - features.shape[0])*[lf], dim=0)
        # print("ft:",idx,lf.size())
        features = torch.cat([features, lf], dim=0)
        return features


    def __getitem__(self, index):
        succ = False
        # while not succ:
        #     try:
        #         feature, label = self.get_feature(index)
        #         succ = True
        #     except Exception as e:
        #         index = self.rng.choice(range(0, self.__len__()))
        #         logging.warning("VideoIter:: ERROR!! (Force using another index:\n{})\n{}".format(index, e))
        feature, label = self.get_feature(index)
        succ = True
        # print("feature: ", feature.size(), "label: ", label)
        return feature, label

    def get_existing_features(self):
        res = []
        for dir in os.listdir(self.features_path):
            dir = os.path.join(self.features_path, dir)
            if os.path.isdir(dir):
                for file in os.listdir(dir):
                    file_no_ext = file.split('.')[0]
                    res.append(os.path.join(dir, file_no_ext))
        return res

    def get_feature(self, index):
        if self.state == 'Normal':  # Load a normal video
            idx = random.randint(0, len(self.features_list_normal) - 1)
            feature_subpath = self.features_list_normal[idx]
            label = 0

        elif self.state == 'Anomalous':  # Load an anomalous video
            idx = random.randint(0, len(self.features_list_anomaly) - 1)
            feature_subpath = self.features_list_anomaly[idx]
            label = 1

        features = read_features(f"{feature_subpath}.txt", self.features_dim, self.bucket_size)

        # print("all features:", features.size())
        if features.shape[0] < self.bucket_size:
            features = self.__padding__(features)
        
        self.state = 'Anomalous' if self.state == 'Normal' else 'Normal'

        return features, label

    @staticmethod
    def _get_features_list(features_path, annotation_path):
        assert os.path.exists(features_path)

        ##### Checking
        # available_features = []
        # abn_classes = os.listdir(features_path)
        # for abn_class_folder in abn_classes:
        #     video_txt = os.listdir(os.path.join(features_path, abn_class_folder))
        #     for video in video_txt:
        #         feat = os.path.join(features_path, abn_class_folder,video)
        #         available_features.append(feat[:-4])
        #         # print(feat[:-4])

        features_list_normal = []
        features_list_anomaly = []
        with open(annotation_path, 'r') as f:
            lines = f.read().splitlines(keepends=False)
            
            for line in lines:
                # items = line.split()
                # file = items[0].split('.')[0]
                # file = file.replace('/', os.sep)
            
                file = line.split('.')[0]
                file = file.replace('/', os.sep)
                
                feature_path = os.path.join(features_path, file)

                # if not feature_path in available_features:
                #     continue

                # print('feature_path: ', feature_path)
                if 'Normal' in feature_path or 'nonviolence' in feature_path or 'NonFight' in feature_path:
                    features_list_normal.append(feature_path)
                else:
                    features_list_anomaly.append(feature_path)

        return features_list_normal, features_list_anomaly

if __name__ == "__main__":
    data = FeaturesLoader(features_path="/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/features2D",
                          annotation_path="/Users/davidchoqueluqueroman/Documents/CODIGOS/AVSS2019/test_ann.txt",
                          bucket_size=30,
                          features_dim=512)
    feature, label = data[0]
    print("feature:", feature.size())
    print("label:", label)

    from torch.utils.data import DataLoader
    loader = DataLoader(data,
                        batch_size=60,
                        shuffle=True,
                        num_workers=1)

    for feature, label in loader:
        print("feature:",feature.size())
        print("label:", label, label.size())