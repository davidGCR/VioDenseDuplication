import os
import sys
sys.path.insert(1, '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/src/VioNet')
# sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/VioDenseDuplication/src')

import json
import numpy as np
import cv2
import visual_utils
# from SORT import Sort
from tube_utils import tube_2_JSON, JSON_2_videoDetections, JSON_2_tube

# from TubeletGeneration.motion_segmentation import MotionSegmentation
# from TubeletGeneration.incremental_linking import IncrementalLinking

from motion_segmentation import MotionSegmentation
from incremental_linking import IncrementalLinking

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
from tube_config import *


def CountFrequency(my_list):
    # Creating an empty dictionary
    freq = {}
    for dcc in my_list:
        # print('erer:',dcc['num'], type(dcc['num']))
        freq[str(dcc['num'])] = freq.get(str(dcc['num']),0) + 1
     
    for key, value in freq.items():
        print ("{} : {}".format(key, value))

def get_videos_from_num_tubes(my_list, num_tubes):
    videos = []
    for dcc in my_list:
        if dcc['num'] == num_tubes:
            videos.append(dcc['path'])
    return videos

def extract_tubes_from_dataset(dataset_persons_detections_path, folder_out, frames):
    
    """
        Args:
            dataset_persons_detections_path: Path to folder containing the person detections in JSON format
    """
    videos_list = os.listdir(dataset_persons_detections_path)
    videos_list = sorted(videos_list)
    num_live_paths = []
    
    for i, video_folder in enumerate(videos_list):
        assert '.json' in video_folder, 'Unrecognized format!!!'
        print("Processing ({}/{}), pt: {}/{} ...".format(i+1,len(videos_list), dataset_persons_detections_path, video_folder))
        
        if not os.path.isdir(folder_out):
            os.makedirs(folder_out)
        
        if os.path.exists(os.path.join(folder_out, video_folder)):
            print('Already done!!!')
            continue

        person_detections = JSON_2_videoDetections("{}/{}".format(dataset_persons_detections_path, video_folder))
        TUBE_BUILD_CONFIG['person_detections'] = person_detections
        segmentator = MotionSegmentation(MOTION_SEGMENTATION_CONFIG)
        tube_builder = IncrementalLinking(TUBE_BUILD_CONFIG)

        live_paths = tube_builder(frames, segmentator, None, False)
        print('live_paths: ', len(live_paths))
        # num_live_paths.append({
        #     'path': dataset_persons_detections_path,
        #     'num': len(live_paths)
        #     })

        tube_2_JSON(output_path=os.path.join(folder_out, video_folder), tube=live_paths)
        
    
        
    # CountFrequency(num_live_paths)

    # videos_no_tubes = get_videos_from_num_tubes(num_live_paths, 0)
    # for v in videos_no_tubes:
    #     print(v)

    return num_live_paths

def extract_tubes_from_video(frames, plot=None):
    segmentator = MotionSegmentation(MOTION_SEGMENTATION_CONFIG)
    tube_builder = IncrementalLinking(TUBE_BUILD_CONFIG)
    live_paths = tube_builder(frames, segmentator, plot, False)
    print('live_paths: ', len(live_paths))
    # for lp in live_paths:
    #     print(lp['score'])
    # CountFrequency([{
    #     'path': '',
    #     'num': len(live_paths)
    #     }])


if __name__=="__main__":
    # vname = "-1l5631l3fg_2"
    # decodedArray = JSON_2_videoDetections("/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/RWF-2000/train/Fight/{}.json".format(vname))
    # decodedArray = JSON_2_videoDetections("/media/david/datos/Violence DATA/PersonDetections/RWF-2000/train/Fight/{}.json".format(vname))
    # print("decodedArray: ", type(decodedArray), len(decodedArray), decodedArray[0])
    # plot_image_detections(decodedArray, "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames")
    
    # dataset_root = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames'
    # dataset_root = '/media/david/datos/Violence DATA/RWF-2000/frames'
    # split = 'train/Fight'
    # video = 'rn7Qjaj9_1'
    # persons_detections_path = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/RWF-2000/{}/{}.json'.format(split,video)
    # person_detections = JSON_2_videoDetections(persons_detections_path)
    # frames = list(range(65,90))
    # extract_tubes_from_video(dataset_root,
    #                         person_detections,
    #                         frames,
    #                         None#{'wait': 1000}
    #                         )


    # tube_2_JSON(output_path=vname+'.json', tube=live_paths)
    # print('Paths ---live_paths[lp][frames_name]=', [lp['frames_name'] for lp in live_paths])
    # print('Paths ---live_paths[lp][frames_name]=', [(len(lp['boxes']), lp['len']) for lp in live_paths])
    # print('Live Paths Final:', len(live_paths))
   
    # tubes = JSON_2_tube('/media/david/datos/Violence DATA/Tubes/RWF-2000/train/Fight/_6-B11R9FJM_2.json')
    # print(len(tubes))
    # print(tubes[0])


    #ONE VIDEO test
    # ucfcrime2local_config = {
    #     'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips',
    #     'split': 'anomaly',
    #     'video': 'Arrest028(2165-2297)',
    #     'p_d_path': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/ucfcrime2local'
    # }
    # config=ucfcrime2local_config

    # rwf_config = {
    #     'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames',
    #     'split': 'train/NonFight',
    #     'video': '1ahhhDBQHxg_0',#'_2RYnSFPD_U_0',
    #     'p_d_path': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/RWF-2000'
    # }
    # config = rwf_config
    # persons_detections_path = config['p_d_path']+'/{}/{}.json'.format(config['split'],config['video'])
    # person_detections = JSON_2_videoDetections(persons_detections_path)
    # frames = np.linspace(0, 149, 25,dtype=np.int16).tolist()
    # # frames = np.linspace(0, 149, dtype=np.int16).tolist()
    # print('random frames: ', frames)

    # TUBE_BUILD_CONFIG['dataset_root'] = config['dataset_root']
    # TUBE_BUILD_CONFIG['person_detections'] = person_detections

    # extract_tubes_from_video(frames,
    #                             {'wait': 1000}
    #                             )
    
    ########################################PROCESS ALL DATASET
    rwf_config = {
        'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames',
        'path_in':'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/RWF-2000',
        'path_out':'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/ActionTubes/RWF-2000-25frames-motion-maps',
        'splits':['train/Fight', 'train/NonFight', 'val/Fight', 'val/NonFight'],
        'start_frame':0,
        'seg_len': 150
    }
    frames = np.linspace(0, 149, 25,dtype=np.int16).tolist()
    config = rwf_config
    TUBE_BUILD_CONFIG['dataset_root'] = config['dataset_root']
    for sp in config['splits']:
        extract_tubes_from_dataset(dataset_persons_detections_path=os.path.join(config['path_in'], sp),
                                    folder_out=os.path.join(config['path_out'], sp),
                                    frames=frames)
    ########################################
    # hockey_config = {
    #     'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/HockeyFightsDATASET/frames',
    #     'path_in':'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/hockey',
    #     'path_out':'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/ActionTubes/hockey2',
    #     'splits':['violence', 'nonviolence'],
    #     'start_frame':0,
    #     'seg_len': 150
    # }
    # frames = np.linspace(0, 39, 25,dtype=np.int16).tolist()
    # config = hockey_config
    # for sp in config['splits']:
    #     extract_tubes_from_dataset(dataset_persons_detections_path=os.path.join(config['path_in'], sp),
    #                                 folder_out=os.path.join(config['path_out'], sp),
    #                                 dataset_root=config['dataset_root'],
    #                                 # start_frame=config['start_frame'],
    #                                 # seg_len=config['seg_len']
    #                                 frames=frames)

