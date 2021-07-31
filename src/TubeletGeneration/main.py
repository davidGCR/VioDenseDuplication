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


def plot_image_detections(decodedArray, dataset_path):
    for item in decodedArray:
        img_path = os.path.join(dataset_path, item['split'], item['video'], item['fname'])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        pred_boxes = item['pred_boxes']
        
        merged_pred_boxes = merge_close_detections(pred_boxes)
        pred_tags_name = item['tags']
        if merged_pred_boxes is not None:
            pred_boxes = np.concatenate((pred_boxes, merged_pred_boxes), axis=0)
            pred_tags_name = pred_tags_name = item['tags'] + ["merged"]*merged_pred_boxes.shape[0]
        
        
        print(item)
        if pred_boxes.shape[0] != 0:
            image = visual_utils.draw_boxes(image,
                                            pred_boxes[:, :4],
                                            scores=pred_boxes[:, 4],
                                            tags=pred_tags_name,
                                            line_thick=1, 
                                            line_color='white')
        name = img_path.split('/')[-1].split('.')[-2]
        # fpath = '{}/{}.png'.format(out_path, name)
        # cv2.imwrite(fpath, image)
        cv2.imshow(name, image)
        key = cv2.waitKey(800)#pauses for 3 seconds before fetching next image
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            # break

# def tracking(decodedArray, 
#             vname,
#             dataset_frames_path="/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames", 
#             video_out_path = '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/videos_',
#             plot=True):

#     mot_tracker = Sort(iou_threshold=0.1)
#     start_tracking = False
#     tracked_objects = None

#     if plot:
#         size = (224,224)
#         result = cv2.VideoWriter('{}/{}.avi'.format(video_out_path,vname), 
#                             cv2.VideoWriter_fourcc('M','J','P','G'),
#                             10, size)


#     for i, item in enumerate(decodedArray): #frame by frame
        
#         if plot:
#             img_path = os.path.join(dataset_frames_path, item['split'], item['video'], item['fname'])
#             image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
#         # print("image shape:", image.shape)

#         pred_boxes = item["pred_boxes"]
#         pred_tags_name = item['tags']

#         if not start_tracking:
#             merge_pred_boxes = merge_close_detections(pred_boxes, only_merged=True)
#             start_tracking = True if len(merge_pred_boxes)>0 else False
#         # pred_boxes = merge_pred_boxes
#         # print("pred_boxes: ", pred_boxes.shape)
        
#         # if merge_pred_boxes is not None:
#             # print("merge_pred_boxes: ", merge_pred_boxes.shape)
        
#         if pred_boxes.shape[0] != 0 and plot:
#             image = visual_utils.draw_boxes(image,
#                                             pred_boxes[:, :4],
#                                             # scores=pred_boxes[:, 4],
#                                             # tags=pred_tags_name,
#                                             line_thick=1, 
#                                             line_color='white')
#         if start_tracking:
#             if tracked_objects is not None: 
#                 if pred_boxes.shape[0] == 0:
#                     pred_boxes = np.empty((0,5))
#                     print('tracked_objects(no persons in frame):frame {}'.format(i+1),tracked_objects.shape)
#                 else:
#                     pred_boxes = merge_close_detections(pred_boxes, only_merged=False) #merge close bboxes in every frame
#                     pred_boxes = np.stack(pred_boxes)
#                     tracked_objects = mot_tracker.update(pred_boxes)
#                     print('tracked_objects:frame {}'.format(i+1),tracked_objects.shape, ' ids: ', tracked_objects[:,4])
#             else:
#                 merge_pred_boxes = np.stack(merge_pred_boxes)
#                 tracked_objects = mot_tracker.update(merge_pred_boxes)
#                 print('--tracked_objects:frame {}'.format(i+1),tracked_objects.shape, ' ids: ', tracked_objects[:,4])      
#             if plot:
#                 image = visual_utils.draw_boxes(image,
#                                                 tracked_objects[:, :4],
#                                                 ids=tracked_objects[:, 4],
#                                                 # tags=["per-"]*tracked_objects.shape[0],
#                                                 line_thick=2, 
#                                                 line_color='green')

#         if plot:
#             result.write(image) # save video
#             name = img_path.split('/')[-1].split('.')[-2]
#             cv2.imshow(name, image)
#             key = cv2.waitKey(10)#pauses for 3 seconds before fetching next image
#             if key == 27:#if ESC is pressed, exit loop
#                 cv2.destroyAllWindows()
#     if plot:
#         result.release()
#         # cv2.destroyAllWindows()
    
#     return tracked_objects


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

def extract_tubes_from_dataset(dataset_persons_detections_path, folder_out, dataset_root, start_frame, seg_len):
    
    """
        Args:
            dataset_persons_detections_path: Path to folder containing the person detections in JSON format
    """
    # start_frame = 60
    frames = list(range(start_frame, start_frame+seg_len))
    videos_list = os.listdir(dataset_persons_detections_path)
    num_live_paths = []
    for i, video_folder in enumerate(videos_list):
        assert '.json' in video_folder, 'Unrecognized format!!!'
        print("Processing ({}/{}), pt: {}/{} ...".format(i+1,len(videos_list), dataset_persons_detections_path, video_folder))
        person_detections = JSON_2_videoDetections("{}/{}".format(dataset_persons_detections_path, video_folder))
        
        segmentator = MotionSegmentation(video_detections=person_detections,
                                            dataset_root=dataset_root)
        tube_builder = IncrementalLinking(video_detections=person_detections,
                                            iou_thresh=0.3,
                                            jumpgap=5,
                                            dataset_root=dataset_root)

        live_paths = tube_builder(frames, segmentator, None)
        print('live_paths: ', len(live_paths))
        num_live_paths.append({
            'path': dataset_persons_detections_path,
            'num': len(live_paths)
            })
        if not os.path.isdir(folder_out):
            os.makedirs(folder_out)
        tube_2_JSON(output_path=os.path.join(folder_out, video_folder), tube=live_paths)
    # CountFrequency(num_live_paths)

    # videos_no_tubes = get_videos_from_num_tubes(num_live_paths, 0)
    # for v in videos_no_tubes:
    #     print(v)

    return num_live_paths

    


def extract_tubes_from_video(dataset_root, persons_detections, frames, plot=None):
    segmentator = MotionSegmentation(video_detections=persons_detections,
                                        dataset_root=dataset_root)
    tube_builder = IncrementalLinking(video_detections=persons_detections,
                                        iou_thresh=0.3,
                                        jumpgap=5,
                                        dataset_root=dataset_root)

    live_paths = tube_builder(frames, segmentator, plot)
    print('live_paths: ', len(live_paths))
    CountFrequency([{
        'path': '',
        'num': len(live_paths)
        }])


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

    ##processing RWF-2000 dataset
    
    # path_in = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/RWF-2000'
    # path_out = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/Tubes/RWF-2000'
   
    # path_in = '/media/david/datos/Violence DATA/PersonDetections/RWF-2000'
    # path_out = '/media/david/datos/Violence DATA/ActionTubes/RWF-2000'
   
    # splits = ['train/Fight', 'train/NonFight', 'val/Fight', 'val/NonFight']
    # for sp in splits:
    #     extract_tubes_from_dataset(dataset_persons_detections_path=os.path.join(path_in, sp),
    #                                 folder_out=os.path.join(path_out, sp),
    #                                 dataset_root=dataset_root)
   
    # tubes = JSON_2_tube('/media/david/datos/Violence DATA/Tubes/RWF-2000/train/Fight/_6-B11R9FJM_2.json')
    # print(len(tubes))
    # print(tubes[0])


    ##ONE VIDEO test
    # ucfcrime2local_config = {
    #     'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips',
    #     'split': 'anomaly',
    #     'video': 'Arrest028(2165-2297)',
    #     'p_d_path': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/ucfcrime2local'
    # }

    rwf_config = {
        'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames',
        'split': 'val/NonFight',
        'video': 'PNgE-OKjhgU_0',
        'p_d_path': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/RWF-2000'
    }
    config = rwf_config
    persons_detections_path = config['p_d_path']+'/{}/{}.json'.format(config['split'],config['video'])
    person_detections = JSON_2_videoDetections(persons_detections_path)
    frames = list(range(50,75))
    extract_tubes_from_video(config['dataset_root'],
                            person_detections,
                            frames,
                            {'wait': 500}
                            )

    ##PROCESS ALL DATASET
    # rwf_config = {
    #     'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames',
    #     'path_in':'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/RWF-2000',
    #     'path_out':'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/ActionTubes/RWF-2000-150frames',
    #     'splits':['train/Fight', 'train/NonFight', 'val/Fight', 'val/NonFight'],
    #     'start_frame':0,
    #     'seg_len': 150
    # }
    # config = rwf_config
    
   
    # for sp in config['splits']:
    #     extract_tubes_from_dataset(dataset_persons_detections_path=os.path.join(config['path_in'], sp),
    #                                 folder_out=os.path.join(config['path_out'], sp),
    #                                 dataset_root=config['dataset_root'],
    #                                 start_frame=config['start_frame'],
    #                                 seg_len=config['seg_len'])

