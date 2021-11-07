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
import time
from utils import natural_sort, TimeMeter

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

def extract_tubes_from_dataset(
    dataset_root, 
    split, 
    person_detections_root, 
    folder_out, 
    frames=None,
    meter=None):
    """
        Args:
            person_detections_root: Path to folder containing the person detections in JSON format
    """
    videos_list = os.listdir(os.path.join(dataset_root, split))
    videos_list = sorted(videos_list)
    num_videos = len(videos_list)
    num_live_paths = []
    

    TUBE_BUILD_CONFIG['dataset_root'] = dataset_root

    folder_out = os.path.join(folder_out, split)
    if not os.path.isdir(folder_out): #Create folder of split
        os.makedirs(folder_out)  

    for i, video in enumerate(videos_list):
        video_path = os.path.join(dataset_root, split, video)
        person_detections_file = os.path.join(person_detections_root, split, video+'.json')
        print("Processing ({}/{}), pt: {}".format(i+1, num_videos, video_path))

        assert '.json' in person_detections_file, 'Unrecognized format!!!'
        
        file_out = os.path.join(folder_out, video+'.json')
        if os.path.exists(file_out):
            print('Already done!!!')
            continue
        person_detections = JSON_2_videoDetections(person_detections_file)

        #video frames path
        if frames == None:
            # frames = np.linspace(0, len(person_detections)-1, len(person_detections), dtype=np.int16).tolist()
            frames_names = os.listdir(video_path)
            frames_names = natural_sort(frames_names)
            # print('\nframes_names: ', frames_names, len(frames_names))
            frames = list(range(0,len(frames_names)))

        TUBE_BUILD_CONFIG['person_detections'] = person_detections
        segmentator = MotionSegmentation(MOTION_SEGMENTATION_CONFIG)
        tube_builder = IncrementalLinking(TUBE_BUILD_CONFIG)

        start = time.time()
        live_paths = tube_builder(frames, frames_names, segmentator)
        end = time.time()
        exc_time = end-start
        print('live_paths: ', len(live_paths))

        meter.update(exc_time, len(frames))
        
        # num_live_paths.append({
        #     'path': dataset_persons_detections_path,
        #     'num': len(live_paths)
        #     })

        tube_2_JSON(output_path=file_out, tube=live_paths)
        frames = None
    
    
        
    # CountFrequency(num_live_paths)

    # videos_no_tubes = get_videos_from_num_tubes(num_live_paths, 0)
    # for v in videos_no_tubes:
    #     print(v)

    return num_live_paths

def extract_tubes_from_video(frames, frames_names, motion_seg_config, tube_build_config, gt=None):
    segmentator = MotionSegmentation(motion_seg_config)
    tube_builder = IncrementalLinking(tube_build_config)
    start = time.time()
    live_paths = tube_builder(frames, frames_names, segmentator, gt)
    end = time.time()
    exec_time = end - start
    return  live_paths, exec_time

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

    # save_folder = os.path.join(
    #     '/Users/davidchoqueluqueroman/Downloads/TubeGenerationExamples',
    #     config['split'],
    #     config['video']
    #     )
    # if not os.path.isdir(save_folder):
    #     os.makedirs(save_folder)

    # save_folder_debug = os.path.join(save_folder, 'debug')
    # if not os.path.isdir(save_folder_debug):
    #     os.makedirs(save_folder_debug)
    # TUBE_BUILD_CONFIG['plot_config']['save_folder_debug'] = save_folder_debug

    TUBE_BUILD_CONFIG['plot_config']['plot_tubes'] = True
    # save_folder_final = os.path.join(save_folder, 'final')
    # if not os.path.isdir(save_folder_final):
    #     os.makedirs(save_folder_final)
    # TUBE_BUILD_CONFIG['plot_config']['save_folder_final'] = save_folder_final

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
        frames,
        MOTION_SEGMENTATION_CONFIG,
        TUBE_BUILD_CONFIG
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

def ucfcrime_one_video_test():
    time_meter = TimeMeter()
    ucf_config = {
        'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime_Reduced/frames',
        'split': 'train/normal',
        'video': 'Normal_Videos108_x264',#'Normal_Videos180_x264',#'V_683',
        'p_d_path': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/UCFCrime_Reduced'
    }
    config = ucf_config
    persons_detections_path = config['p_d_path']+'/{}/{}.json'.format(config['split'],config['video'])
    person_detections = JSON_2_videoDetections(persons_detections_path)
    # frames = np.linspace(0, len(person_detections)-1, num=len(person_detections), dtype=np.int16).tolist()
    # frames = np.linspace(0, 149, dtype=np.int16).tolist()
    
    frames_names = os.listdir(os.path.join(ucf_config['dataset_root'], ucf_config['split'], ucf_config['video']))
    frames_names = natural_sort(frames_names)
    print('frames_names: ', frames_names, len(frames_names))
    frames = list(range(0,len(frames_names)))

    print('random frames: ', frames, len(frames))

    TUBE_BUILD_CONFIG['dataset_root'] = config['dataset_root']
    TUBE_BUILD_CONFIG['person_detections'] = person_detections

    plot_create_save_dirs(config)

    live_paths, exec_time= extract_tubes_from_video(
        frames,
        frames_names,
        MOTION_SEGMENTATION_CONFIG,
        TUBE_BUILD_CONFIG
        )
    clip_len = len(frames)
    num_runs = 1
    frame_rate = 30

    FPS = clip_len*num_runs/exec_time
    print('\nlive_paths: {}, time: {} seconds, FPS: {}'.format(len(live_paths), exec_time, FPS))

    time_meter.update(exec_time, clip_len)

    print('TimeMeter FPS ---> time: {}, FPS: {}'.format(time_meter.total_time, time_meter.fps))

    for i, lp in enumerate(live_paths):
        print('\n lp:{}'.format(i+1))
        print(lp)

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
    # rwf_config = {
    #     'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames',
    #     'path_in':'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/RWF-2000',
    #     'path_out':'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/ActionTubes/final/rwf',
    #     'splits':['train/Fight', 'train/NonFight', 'val/Fight', 'val/NonFight'],
    #     'start_frame':0,
    #     'seg_len': 150
    # }
    # frames = np.linspace(0, 149,dtype=np.int16).tolist()
    # config = rwf_config
    # TUBE_BUILD_CONFIG['dataset_root'] = config['dataset_root']
    # for sp in config['splits']:
    #     extract_tubes_from_dataset(dataset_persons_detections_path=os.path.join(config['path_in'], sp),
    #                                 folder_out=os.path.join(config['path_out'], sp),
    #                                 frames=frames)
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

    ############################################################################################################
    # ucfcrime_one_video_test()
    ucf_config = {
        'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime_Reduced/frames',
        'path_in':'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/UCFCrime_Reduced',
        'path_out':'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/ActionTubesV2/UCFCrime_Reduced',
        'splits':['train/abnormal', 'train/normal', 'test/abnormal', 'test/normal']
    }
    frames = None
    config = ucf_config
    # TUBE_BUILD_CONFIG['dataset_root'] = config['dataset_root']
    time_meter = TimeMeter()
    for sp in config['splits']:
        extract_tubes_from_dataset(dataset_root=config['dataset_root'],
                                    split=sp,
                                    person_detections_root=config['path_in'],
                                    folder_out=config['path_out'],
                                    frames=frames,
                                    meter=time_meter)
    print('TimeMeter FPS ---> time: {}, FPS: {}'.format(time_meter.total_time, time_meter.fps))
