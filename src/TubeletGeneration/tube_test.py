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

def extract_tubes_from_dataset(dataset_persons_detections_path, folder_out, frames=None):
    
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
        if frames == None:
            frames = np.linspace(0, len(person_detections)-1, dtype=np.int16).tolist()
        TUBE_BUILD_CONFIG['person_detections'] = person_detections
        segmentator = MotionSegmentation(MOTION_SEGMENTATION_CONFIG)
        tube_builder = IncrementalLinking(TUBE_BUILD_CONFIG)

        live_paths = tube_builder(frames, segmentator)
        print('live_paths: ', len(live_paths))
        
        # num_live_paths.append({
        #     'path': dataset_persons_detections_path,
        #     'num': len(live_paths)
        #     })

        tube_2_JSON(output_path=os.path.join(folder_out, video_folder), tube=live_paths)
        frames = None
    
        
    # CountFrequency(num_live_paths)

    # videos_no_tubes = get_videos_from_num_tubes(num_live_paths, 0)
    # for v in videos_no_tubes:
    #     print(v)

    return num_live_paths

def extract_tubes_from_video(frames, motion_seg_config, tube_build_config, gt=None):
    # segmentator = MotionSegmentation(MOTION_SEGMENTATION_CONFIG)
    # tube_builder = IncrementalLinking(TUBE_BUILD_CONFIG)
    segmentator = MotionSegmentation(motion_seg_config)
    tube_builder = IncrementalLinking(tube_build_config)
    live_paths = tube_builder(frames, segmentator, gt)
    return  live_paths

def plot_video_tube(tube_json):
    tubes = JSON_2_tube(tube_json)
    print(tubes)
    video_folder_path = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames'
    video_name = tube_json.split('/')[-1][:-5]

    video_folder_path = os.path.join(video_folder_path,'train/Fight',video_name)
    one_tube = tubes[0]
    for i in range(len(one_tube['frames_name'])):
        frame_path = os.path.join(video_folder_path, one_tube['frames_name'][i])
        frame_box = one_tube['boxes'][i]
        image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        print('iamge shape: ', image.shape)
        x1,y1,x2,y2 = np.array(frame_box[:4]).astype(int)
        cv2.rectangle(image,
                    (x1, y1),
                    (x2, y2),
                    (255,0,0),
                    2)
        cv2.imshow(one_tube['frames_name'][i], image)
        key = cv2.waitKey(100)#pauses for 3 seconds before fetching next image
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows()

def plot_create_save_dirs(config):
    TUBE_BUILD_CONFIG['plot_config']['debug_mode'] = False
    TUBE_BUILD_CONFIG['plot_config']['plot_wait_tubes'] = 1000
    TUBE_BUILD_CONFIG['plot_config']['plot_wait_2'] = 20
    TUBE_BUILD_CONFIG['plot_config']['save_results'] = False

    save_folder = os.path.join(
        '/Users/davidchoqueluqueroman/Downloads/TubeGenerationExamples',
        config['split'],
        config['video']
        )
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    save_folder_debug = os.path.join(save_folder, 'debug')
    if not os.path.isdir(save_folder_debug):
        os.makedirs(save_folder_debug)
    TUBE_BUILD_CONFIG['plot_config']['save_folder_debug'] = save_folder_debug

    TUBE_BUILD_CONFIG['plot_config']['plot_tubes'] = True
    save_folder_final = os.path.join(save_folder, 'final')
    if not os.path.isdir(save_folder_final):
        os.makedirs(save_folder_final)
    TUBE_BUILD_CONFIG['plot_config']['save_folder_final'] = save_folder_final

def rwf_one_video_test():
    rwf_config = {
        'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames',
        'split': 'train/Fight',
        'video': 'u1r8f71c_3',#'dt8YUGoOSgQ_0',#'C8wt47cphU8_1',#'_2RYnSFPD_U_0',
        'p_d_path': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/RWF-2000'
    }
    config = rwf_config
    persons_detections_path = config['p_d_path']+'/{}/{}.json'.format(config['split'],config['video'])
    person_detections = JSON_2_videoDetections(persons_detections_path)
    frames = np.linspace(0, 149,dtype=np.int16).tolist()
    # frames = np.linspace(0, 149, dtype=np.int16).tolist()
    print('random frames: ', frames)

    TUBE_BUILD_CONFIG['dataset_root'] = config['dataset_root']
    TUBE_BUILD_CONFIG['person_detections'] = person_detections

    # plot_create_save_dirs(config)
    TUBE_BUILD_CONFIG['plot_config']['plot_tubes'] = True
    TUBE_BUILD_CONFIG['plot_config']['debug_mode'] = True

    live_paths = extract_tubes_from_video(
        frames
        )
    
    print('live_paths: ', len(live_paths))
    print(live_paths[0])

def hockey_one_video_test():
    hockey_config = {
        'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/HockeyFightsDATASET/frames',
        'split': 'violence',
        'video': '320',
        'p_d_path': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/hockey'
    }
    config = hockey_config
    persons_detections_path = config['p_d_path']+'/{}/{}.json'.format(config['split'],config['video'])
    person_detections = JSON_2_videoDetections(persons_detections_path)
    frames = np.linspace(0, 39,dtype=np.int16).tolist()
    # frames = np.linspace(0, 149, dtype=np.int16).tolist()
    print('random frames: ', frames)

    plot_create_save_dirs(config)

    TUBE_BUILD_CONFIG['dataset_root'] = config['dataset_root']
    TUBE_BUILD_CONFIG['person_detections'] = person_detections

    live_paths = extract_tubes_from_video(
        frames
        )
    
    print('live_paths: ', len(live_paths))

def rlvs_one_video_test():
    hockey_config = {
        'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RealLifeViolenceDataset/frames',
        'split': 'Violence',
        'video': 'V_683',#'V_683',
        'p_d_path': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/RealLifeViolenceDataset'
    }
    config = hockey_config
    persons_detections_path = config['p_d_path']+'/{}/{}.json'.format(config['split'],config['video'])
    person_detections = JSON_2_videoDetections(persons_detections_path)
    frames = np.linspace(0, len(person_detections)-1,dtype=np.int16).tolist()
    # frames = np.linspace(0, 149, dtype=np.int16).tolist()
    print('random frames: ', frames)

    TUBE_BUILD_CONFIG['dataset_root'] = config['dataset_root']
    TUBE_BUILD_CONFIG['person_detections'] = person_detections

    plot_create_save_dirs(config)

    live_paths = extract_tubes_from_video(
        frames
        )
    
    print('live_paths: ', len(live_paths))

if __name__=="__main__":
    # plot_video_tube('/Users/davidchoqueluqueroman/Documents/DATASETS_Local/ActionTubes/RWF-2000-150frames-motion-maps2/train/Fight/OAfV0xPIhZw_2.json')
    #ONE VIDEO test
    # ucfcrime2local_config = {
    #     'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips',
    #     'split': 'anomaly',
    #     'video': 'Arrest028(2165-2297)',
    #     'p_d_path': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/ucfcrime2local'
    # }
    # config=ucfcrime2local_config
    ########################################ONE VIDEO
    # rwf_one_video_test()
    # hockey_one_video_test()
    
    # rlvs_one_video_test()
    ########################################PROCESS ALL DATASET
    rwf_config = {
        'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames',
        'path_in':'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/RWF-2000',
        'path_out':'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/ActionTubes/final/rwf',
        'splits':['train/Fight', 'train/NonFight', 'val/Fight', 'val/NonFight'],
        'start_frame':0,
        'seg_len': 150
    }
    frames = np.linspace(0, 149,dtype=np.int16).tolist()
    config = rwf_config
    TUBE_BUILD_CONFIG['dataset_root'] = config['dataset_root']
    for sp in config['splits']:
        extract_tubes_from_dataset(dataset_persons_detections_path=os.path.join(config['path_in'], sp),
                                    folder_out=os.path.join(config['path_out'], sp),
                                    frames=frames)
    ############################################################################################################
    # hockey_config = {
    #     'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/HockeyFightsDATASET/frames',
    #     'path_in':'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/hockey',
    #     'path_out':'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/ActionTubes/final/hockey',
    #     'splits':['violence', 'nonviolence'],
    #     'start_frame':0,
    #     'seg_len': 150
    # }
    # frames = np.linspace(0, 39,dtype=np.int16).tolist()
    # config = hockey_config
    # TUBE_BUILD_CONFIG['dataset_root'] = config['dataset_root']
    # for sp in config['splits']:
    #     extract_tubes_from_dataset(dataset_persons_detections_path=os.path.join(config['path_in'], sp),
    #                                 folder_out=os.path.join(config['path_out'], sp),
    #                                 frames=frames)

    ############################################################################################################
    # rlvs_config = {
    #     'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RealLifeViolenceDataset/frames',
    #     'path_in':'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/RealLifeViolenceDataset',
    #     'path_out':'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/ActionTubes/RealLifeViolenceDataset',
    #     'splits':['Violence', 'NonViolence']
    # }
    # frames = None
    
    # config = rlvs_config
    # TUBE_BUILD_CONFIG['dataset_root'] = config['dataset_root']
    # for sp in config['splits']:
        
    #     extract_tubes_from_dataset(dataset_persons_detections_path=os.path.join(config['path_in'], sp),
    #                                 folder_out=os.path.join(config['path_out'], sp),
    #                                 frames=frames)

