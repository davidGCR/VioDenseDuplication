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
                 features_dim=4096,
                 metadata=False,
                 shuffle=True):

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
        self.metadata = metadata
        
        if shuffle:
            self.shuffle()
        # else:
        #     print("hereeeeeeee")
        #     self.features_list_normal.sort()
        #     self.features_list_anomaly.sort()

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
        feature, label, feature_subpath, idx = self.get_feature()
        succ = True
        # print("feature: ", feature.size(), "label: ", label)
        if self.metadata:
            return feature, label, feature_subpath, idx
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

    def get_feature(self):
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

        return features, label, feature_subpath, idx

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
                items = line.split()
                file = items[0].split('.')[0]
                file = file.replace('/', os.sep)
            
                # file = line.split('.')[0]
                # file = file.replace('/', os.sep)
                
                feature_path = os.path.join(features_path, file)

                # if not feature_path in available_features:
                #     continue

                # print('feature_pathhhhh: ', feature_path)
                if 'Normal' in feature_path or 'nonviolence' in feature_path or 'NonFight' in feature_path:
                    features_list_normal.append(feature_path)
                else:
                    features_list_anomaly.append(feature_path)

        return features_list_normal, features_list_anomaly

class ConcatFeaturesLoader(Dataset):
    def __init__(self,
                 features_path_1,
                 features_path_2,
                 annotation_path,
                 bucket_size,
                 features_dim_1,
                 features_dim_2,
                 metadata=True):
        self.FeaturesLoader_1 = FeaturesLoader(features_path=features_path_1,
                                               annotation_path=annotation_path,
                                               bucket_size=bucket_size,
                                               features_dim=features_dim_1,
                                               metadata=metadata,
                                               shuffle=False)
        # self.FeaturesLoader_2 = FeaturesLoader(features_path=features_path_2, annotation_path=annotation_path, bucket_size=bucket_size, features_dim=features_dim_2, metadata=metadata, shuffle=False)
        # self.metadata = metadata
        self.features_dim_2 = features_dim_2
        self.bucket_size = bucket_size
        self.features_list_normal, self.features_list_anomaly = self.FeaturesLoader_1._get_features_list(features_path=features_path_2,
                                                                                                    annotation_path=annotation_path)
    
    def __len__(self):
        return self.bucket_size*2
    
    def get_feature(self, label, idx):
        if label == 0:
            feature_subpath = self.features_list_normal[idx]
        else:
            feature_subpath = self.features_list_anomaly[idx]
        features = read_features(f"{feature_subpath}.txt", self.features_dim_2, self.bucket_size)
        if features.shape[0] < self.bucket_size:
            features = self.FeaturesLoader_1.__padding__(features)
        return features, feature_subpath

    
    def __getitem__(self, index):
        feature_1, label_1, feature_subpath_1, idx = self.FeaturesLoader_1[index]
        feature_2, feature_subpath_2 = self.get_feature(label_1, idx)
        # print("idx:", idx)
        # print(' feature_subpath_1:',  feature_subpath_1)
        # print(' feature_subpath_2:',  feature_subpath_2)
        

        feature_concat = torch.cat([feature_1, feature_2], dim=1)
        return feature_concat, label_1

if __name__ == "__main__":
    data_combined = ConcatFeaturesLoader(features_path_1="/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/features_input(dynamic-images)_frames(16)",
                                         features_path_2="/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/features_S3D_input(rgb)_frames(16)",
                                         annotation_path="/Users/davidchoqueluqueroman/Documents/CODIGOS/AVSS2019/ucfcrime2local_train_ann.txt",
                                         bucket_size=30,
                                         features_dim_1=512,
                                         features_dim_2=1024,
                                         metadata=True)
    # feature, label = data_combined[44]
    # print("label_1:", label)
    # print("feature:", feature.size())

    from torch.utils.data import DataLoader
    loader = DataLoader(data_combined,batch_size=60,shuffle=True,num_workers=1)
    
    for feature, label in loader:
        print("feature:",feature.size())
        print("label:", label, label.size())

    # data = FeaturesLoader(features_path="/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/features2D",
    #                       annotation_path="/Users/davidchoqueluqueroman/Documents/CODIGOS/AVSS2019/ucfcrime2local_train_ann.txt",
    #                       bucket_size=30,
    #                       features_dim=512)
    # feature, label = data[0]
    # print("feature:", feature.size())
    # print("label:", label)

    # from torch.utils.data import DataLoader
    # loader = DataLoader(data,
    #                     batch_size=60,
    #                     shuffle=True,
    #                     num_workers=1)

    # for feature, label in loader:
    #     print("feature:",feature.size())
    #     print("label:", label, label.size())