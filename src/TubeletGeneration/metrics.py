import os
import sys
sys.path.insert(1, '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/src')
sys.path.insert(1, '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/src/VioNet')

from VioNet.customdatasets.make_dataset import MakeUCFCrime2LocalClips
from TubeletGeneration.motion_segmentation import MotionSegmentation
from TubeletGeneration.incremental_linking import IncrementalLinking
from TubeletGeneration.tube_utils import JSON_2_videoDetections
from collections import Counter
import cv2
import re

def extract_tubes_from_video(dataset_root, persons_detections, frames, plot=None):
    segmentator = MotionSegmentation(video_detections=persons_detections,
                                        dataset_root=dataset_root,
                                        ratio_box_mmap=0.3,
                                        size=5,
                                        segment_size=10,
                                        stride=1,
                                        overlap=0)
    tube_builder = IncrementalLinking(video_detections=persons_detections,
                                        iou_thresh=0.3,
                                        jumpgap=10,
                                        dataset_root=dataset_root)

    live_paths = tube_builder(frames, segmentator, plot)
    print('live_paths: ', len(live_paths))
    for lp in live_paths:
        print(lp['score'])
    return live_paths
    

if __name__=='__main__':
    #ONE VIDEO test
    ucfcrime2local_config = {
        'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips',
        'split': 'anomaly',
        'video': 'Arrest036(2917-3426)',
        'p_d_path': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/ucfcrime2local',
        'gt_ann_path': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CrimeViolence2LocalDATASET/Txt annotations-longVideos'
    }
    config=ucfcrime2local_config
    persons_detections_path = config['p_d_path']+'/{}/{}.json'.format(config['split'],config['video'])
    person_detections = JSON_2_videoDetections(persons_detections_path)
    
    video_len = len(os.listdir(os.path.join(config['dataset_root'], config['split'], config['video'])))
    video_len = 50
    start = 0#60
    split_len = 50
    s=0
    lps=[]
    for i in range(start,start + video_len, split_len):
        print('++++++split: ', s+1)
        frames = list(range(i,i+split_len))
        lps_split = extract_tubes_from_video(config['dataset_root'],
                                person_detections,
                                frames,
                                {'wait': 200}
                                )
        lps += lps_split
    
    
    m = MakeUCFCrime2LocalClips(root='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips',
                                # root_normal='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/frames',
                                path_annotations='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CrimeViolence2LocalDATASET/Txt annotations-longVideos',
                                abnormal=True)
    paths, labels, annotations = m()
    print('paths: ', paths)
    for idx in range(len(paths)):
        if config['video'] in paths[idx].split('/'):
            anns = m.ground_truth_boxes(paths[idx],annotations[idx])
            m.plot(paths[idx], anns, lps)
            break
    # idx=65
    # print(idx)
    # print(Counter(labels))
    # print(paths[idx])
    # print(labels[idx])
    # print(annotations[idx])

    # anns = m.ground_truth_boxes(paths[idx],annotations[idx])
    # m.plot(paths[idx], anns)
    