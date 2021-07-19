import os
from os import path, mkdir
import numpy as np

def to_segments(data, num=32):
    """
    These code is taken from:
    https://github.com/rajanjitenpatel/C3D_feature_extraction/blob/b5894fa06d43aa62b3b64e85b07feb0853e7011a/extract_C3D_feature.py#L805
    :param data: list of features of a certain video
    :return: list of 32 segments
    """
    data = np.array(data)
    Segments_Features = []
    thirty2_shots = np.round(np.linspace(0, len(data) - 1, num=num+1)).astype(int)
    for ss, ee in zip(thirty2_shots[:-1], thirty2_shots[1:]):
        if ss == ee:
            temp_vect = data[min(ss, data.shape[0] - 1), :]
        else:
            temp_vect = data[ss:ee, :].mean(axis=0)

        temp_vect = temp_vect / np.linalg.norm(temp_vect)
        if np.linalg.norm == 0:
            logging.error("Feature norm is 0")
            exit()
        if len(temp_vect) != 0:
            Segments_Features.append(temp_vect.tolist())

    return Segments_Features

class FeaturesWriter:
    def __init__(self, num_videos, num_segments):
        self.path = None
        self.dir = None
        self.data = None
        # self.chunk_size = chunk_size
        self.num_videos = num_videos
        self.dump_count = 0
        self.num_segments = num_segments

    def _init_video(self, video_name, dir):
        self.path = path.join(dir, f"{video_name}.txt")
        self.dir = dir
        self.data = dict()

    def has_video(self):
        return self.data is not None

    def dump(self):
        # logging.info(f'{self.dump_count} / {self.num_videos}:	Dumping {self.path}')
        print(f'{self.dump_count} / {self.num_videos}:	Dumping {self.path} with {len(self.data)} features')
        self.dump_count += 1
        if not path.exists(self.dir):
            os.mkdir(self.dir)

        if self.num_segments>0:
            features = to_segments([self.data[key] for key in sorted(self.data)])
        else:
            features = [self.data[key] for key in sorted(self.data)]
        with open(self.path, 'w') as fp:
            for d in features:
                d = [str(x) for x in d]
                fp.write(' '.join(d) + '\n')

    def _is_new_video(self, video_name, dir):
        new_path = path.join(dir, f"{video_name}.txt")
        if self.path != new_path and self.path is not None:
            return True

        return False

    def store(self, feature, idx):
        self.data[idx] = list(feature)

    def write(self, feature, video_name, idx, dir):
        if not self.has_video():
            self._init_video(video_name, dir)

        if self._is_new_video(video_name, dir):
            self.dump()
            self._init_video(video_name, dir)

        self.store(feature, idx)