import os
import sys
g_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print('main g_path:', g_path)
sys.path.insert(1, g_path)

from VioNet.customdatasets.make_dataset import MakeUCFCrime2Local
import re
import shutil

def extract_frames(path, label, intervals, out_path):
    frames = os.listdir(path)
    frames = [name for name in frames if "jpg" in name]
    
    _, video_name = os.path.split(path)
    # if not os.path.isdir(os.path.join(out_path, video_name)):
    #     os.mkdir(os.path.join(out_path, video_name))
    
    for f_path in [os.path.join(path, f) for f in frames]:
        _, f_name = os.path.split(f_path)
        f_name_int = int(re.search(r'\d+', f_name).group())
        for s,e in intervals:
            split_folder = os.path.join(out_path, video_name+"({}-{})".format(s,e))
            if not os.path.isdir(split_folder):
                os.mkdir(split_folder)
            if f_name_int >= s and f_name_int <= e:
                # print(f_name)
                # it.append(f_name)
                shutil.copy(f_path, os.path.join(split_folder,f_name))


if __name__=="__main__":
    m = MakeUCFCrime2Local(root='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/frames',
                            annotation_path='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/readme',
                            bbox_path='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/readme/Txt annotations',
                            train=True)
    paths, labels, annotations, intervals = m()
    # print(paths, len(paths))
    # print(labels, len(labels))
    # print(Counter(labels))
    # idx=2
    # print(paths[idx])
    # print(labels[idx])
    # print(annotations[idx][0:10])
    # print(intervals[idx])

    for path, label, interval in zip(paths, labels, intervals):
        if label == 1:
            print(path)
            print(label)
            print(interval)
            extract_frames(path, label, interval, "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips")
    