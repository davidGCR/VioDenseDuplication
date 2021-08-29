import os
import sys
sys.path.insert(1, '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/src')
sys.path.insert(1, '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/src/VioNet')

from VioNet.customdatasets.make_dataset import MakeUCFCrime2LocalClips
from TubeletGeneration.motion_segmentation import MotionSegmentation
from TubeletGeneration.incremental_linking import IncrementalLinking, bbox_iou_numpy
from TubeletGeneration.tube_utils import JSON_2_videoDetections
from collections import Counter
import cv2
import re
import numpy as np

from main import extract_tubes_from_video
from tube_config import *

if __name__=='__main__':
    #ONE VIDEO test
    # ucfcrime2local_config = {
    #     'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips',
    #     'split': 'anomaly',
    #     'video': 'Arrest036(2917-3426)',
    #     'p_d_path': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/ucfcrime2local',
    #     'gt_ann_path': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CrimeViolence2LocalDATASET/Txt annotations-longVideos'
    # }
    # config=ucfcrime2local_config
    # persons_detections_path = config['p_d_path']+'/{}/{}.json'.format(config['split'],config['video'])
    # person_detections = JSON_2_videoDetections(persons_detections_path)
    
    # video_len = len(os.listdir(os.path.join(config['dataset_root'], config['split'], config['video'])))
    # video_len = 50
    # start = 0#60
    # split_len = 50
    # s=0
    # lps=[]
    # for i in range(start,start + video_len, split_len):
    #     print('++++++split: ', s+1)
    #     frames = list(range(i,i+split_len))
    #     lps_split = extract_tubes_from_video(config['dataset_root'],
    #                             person_detections,
    #                             frames,
    #                             {'wait': 200}
    #                             )
    #     lps += lps_split
    
    
    # m = MakeUCFCrime2LocalClips(root='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips',
    #                             # root_normal='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/frames',
    #                             path_annotations='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CrimeViolence2LocalDATASET/Txt annotations-longVideos',
    #                             abnormal=True)
    # paths, labels, annotations = m()
    # print('paths: ', paths)
    # for idx in range(len(paths)):
    #     if config['video'] in paths[idx].split('/'):
    #         anns = m.ground_truth_boxes(paths[idx],annotations[idx])
    #         m.plot(paths[idx], anns, lps)
    #         break
    # # anns = m.ground_truth_boxes(paths[idx],annotations[idx])
    # # m.plot(paths[idx], anns)

    from torchvision import transforms
    from VioNet.customdatasets.ucfcrime2local_dataset import UCFCrime2LocalVideoDataset
    #### videos
    # video = 'Stealing091(245-468)'#--224'
    video = 'Arrest002(257-536)'
    video_dataset = UCFCrime2LocalVideoDataset(
        path='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips/anomaly/{}'.format(video),
        sp_annotation='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CrimeViolence2LocalDATASET/Txt annotations-longVideos/{}.txt'.format(video.split('(')[0]),
        p_detections='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/ucfcrime2local/anomaly/{}.json'.format(video),
        transform=transforms.ToTensor(),
        clip_len=280,
        clip_temporal_stride=5
    )

    def loc_error_tube_gt(tube, gt, threshold=0.5):
        fails = 0
        for box_gt in gt:
            frame_number_gt = box_gt['frame']
            print('frame_number_gt:', frame_number_gt)
            # print('tube[frames_name]:', tube['frames_name'])
            f_numbers = [int(re.findall(r'\d+', f_name)[0]) for f_name in tube['frames_name']]
            f_numbers.sort()
            # print('f_numbers', f_numbers)
            if frame_number_gt in  f_numbers:
                for i in range(len(f_numbers)):
                    if f_numbers[i] == frame_number_gt:
                        b1 = np.array([int(box_gt['xmin']), int(box_gt['ymin']), int(box_gt['xmax']), int(box_gt['ymax'])]).reshape((1,4))#(1,4)
                        b2 = np.array(tube['boxes'][i][:4]).reshape((1,4))#(1,4)
                        # print('f_numbers[i] == frame_number_gt', f_numbers[i], frame_number_gt)
                        iou = bbox_iou_numpy(b1,b2) 
                        print('iou(b1: {}, b2: {}) = {}'.format(b1, b2, iou))
                        if iou < threshold:
                            fails += 1
            else:
                fails += 1
        print('fails={}, num_frames={}'.format(fails, len(gt)))
        return fails/len(gt)
            
    
    person_detections = JSON_2_videoDetections(video_dataset.p_detections)
    TUBE_BUILD_CONFIG['dataset_root'] = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips'
    TUBE_BUILD_CONFIG['person_detections'] = person_detections

    for clip, frames, gt in video_dataset:
        print('--',clip, len(clip), frames.size())
        # for g in gt:
        #     print(g)
        
        person_detections = JSON_2_videoDetections(video_dataset.p_detections)
        lps_split = extract_tubes_from_video(
                                clip,
                                # {'wait': 100}
                                )
        le = loc_error_tube_gt(lps_split[0], gt)
        print('localization error: ', le)

        # for lp in lps_split:
        #     iou_tube_gt(lp, gt)
        