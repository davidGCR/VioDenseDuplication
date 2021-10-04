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
import matplotlib
from tube_test import extract_tubes_from_video
from tube_config import *

from torchvision import transforms
from VioNet.utils import get_number_from_string
from sklearn import metrics

def precision_recall_curve(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]

        precision = metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        recall = metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls

def loc_error_tube_gt(tube, gt, threshold=0.5):
    fails = 0
    for k,box_gt in enumerate(gt):
        frame_number_gt = box_gt['frame']
        print(k+1, ' frame_number_gt:', frame_number_gt)
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
                    print('iou([{}]: {}, [{}]: {}) = {}'.format(box_gt['frame'], b1, tube['frames_name'][i], b2, iou))
                    if iou < threshold:
                        fails += 1
        else:
            fails += 1
    print('fails={}, num_frames={}'.format(fails, len(gt)))
    return fails/len(gt)

def st_iou(tube, gt, threshold=0.5):
    tgb = gt[0]['frame'] #gt begin
    tge = gt[-1]['frame'] #gt end

    tdb = get_number_from_string(tube['frames_name'][0]) #tube begin
    tde = get_number_from_string(tube['frames_name'][-1]) #tube end

    # fname = get_number_from_string(tube['frames_name'][0])
    # print('tube len: {}/ {}'.format(tube['frames_name'],tube['len']))

    # print(gt)

    T_i = max(0,min(tge,tde)-max(tgb,tdb))
    # print('min(tge,tde): (tge,tde)=({},{}) = {}'.format(tge, tde, min(tge,tde)))
    # print('max(tgb,tdb): (tgb,tdb)=({},{}) = {}'.format(tgb, tdb, max(tgb,tdb)))
    # print('T_i: {}'.format(T_i))
    if T_i>0:
        T_u = max(tge, tde) - min(tgb, tdb) + 1
        # print('T_u: {}'.format(T_u))
        T_iou = T_i/T_u
        # print('T_iou: {}'.format(T_iou))
        frames_gt = [box_gt['frame'] for box_gt in gt]
        frames_dt = [int(get_number_from_string(f_name)) for f_name in tube['frames_name']]

        int_frames_numb = list(range(max(tgb,tdb), min(tge,tde)+1))
        # print('frames_gt: {}/{}'.format(frames_gt, len(frames_gt)))
        # print('frames_dt: {}/{}'.format(frames_dt, len(frames_dt)))
        # print('int_frames_numb: {}/{}'.format(int_frames_numb, len(int_frames_numb)))

        mask_gt = np.isin(np.array(frames_gt), np.array(int_frames_numb))
        int_find_gt = np.nonzero(mask_gt)[0]
        # print('int_find_gt: {}/{}'.format(int_find_gt, int_find_gt.shape))
        
        mask_dt = np.isin(np.array(frames_dt), np.array(int_frames_numb))
        int_find_dt = np.nonzero(mask_dt)[0]
        # print('int_find_dt: {}/{}'.format(int_find_dt, int_find_dt.shape))

        # assert int_find_gt.shape[0] == int_find_dt.shape[0], 'Error!!!'
        # ious = np.zeros(int_find_dt.shape[0])
        # for i in range(int_find_dt.shape[0]):
        #     gt_frame_name = frames_gt[int_find_gt[i]]
        #     dt_frame_name = frames_dt[int_find_dt[i]]
        #     assert gt_frame_name == dt_frame_name, 'Error: gt and dt inconsistency!!!-->{}!={}'.format(gt_frame_name, dt_frame_name)
        #     b1 = np.array([
        #         int(gt[int_find_gt[i]]['xmin']), 
        #         int(gt[int_find_gt[i]]['ymin']), 
        #         int(gt[int_find_gt[i]]['xmax']), 
        #         int(gt[int_find_gt[i]]['ymax'])]).reshape((1,4))#(1,4)
        #     b2 = np.array(tube['boxes'][int_find_dt[i]][:4]).reshape((1,4))
        #     ious[i] = bbox_iou_numpy(b1,b2)
        ious = np.zeros(len(frames_dt))
        for i in range(len(frames_dt)):
            # gt_frame_name = frames_gt[int_find_gt[i]]
            dt_frame_name = frames_dt[i]
            # assert gt_frame_name == dt_frame_name, 'Error: gt and dt inconsistency!!!-->{}!={}'.format(gt_frame_name, dt_frame_name)
            # print('---processing detected frame:', dt_frame_name)
            if dt_frame_name in frames_gt:
                idx = frames_gt.index(dt_frame_name)
                b1 = np.array([
                    int(gt[idx]['xmin']), 
                    int(gt[idx]['ymin']), 
                    int(gt[idx]['xmax']), 
                    int(gt[idx]['ymax'])]).reshape((1,4))#(1,4)
                b2 = np.array(tube['boxes'][int_find_dt[i]][:4]).reshape((1,4))
                ious[i] = bbox_iou_numpy(b1,b2)
            else:
                ious[i] = 0
        st_iou_ = T_iou*np.mean(ious)
        return st_iou_
    else:
        return 0


def test_one_video(video_dataset):
    from VioNet.customdatasets.ucfcrime2local_dataset import UCFCrime2LocalVideoDataset
    #### videos
    # video = 'Stealing091(245-468)'#--224'
    video = 'Arrest003(1435-2089)' #655 
    # video = 'Arrest005(282-890)' #
    # video = 'Arrest025(334-701)' #368
    # video = 'Assault018(52-71)'  #20
    # video = 'Assault049(377-426)' #50
    video_dataset = UCFCrime2LocalVideoDataset(
        path='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips/anomaly/{}'.format(video),
        sp_annotation='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CrimeViolence2LocalDATASET/Txt annotations-longVideos/{}.txt'.format(video.split('(')[0]),
        p_detections='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/ucfcrime2local/anomaly/{}.json'.format(video),
        transform=transforms.ToTensor(),
        clip_len=654,
        clip_temporal_stride=5
    )

    person_detections = JSON_2_videoDetections(video_dataset.p_detections)
    TUBE_BUILD_CONFIG['dataset_root'] = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips'
    TUBE_BUILD_CONFIG['person_detections'] = person_detections

    TUBE_BUILD_CONFIG['plot_config']['debug_mode'] = False

    TUBE_BUILD_CONFIG['plot_config']['plot_tubes'] = True
    TUBE_BUILD_CONFIG['plot_config']['plot_wait_tubes'] = 100
    save_folder = os.path.join(
        '/Users/davidchoqueluqueroman/Downloads/TubeGenerationExamples',
        video
        )
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    save_folder_final = os.path.join(save_folder, 'final')
    if not os.path.isdir(save_folder_final):
        os.makedirs(save_folder_final)
    TUBE_BUILD_CONFIG['plot_config']['save_folder_final'] = save_folder_final

    
    for clip, frames, gt in video_dataset:
        print('--',clip, len(clip), frames.size())
        for g in gt:
            print(g)
        
        person_detections = JSON_2_videoDetections(video_dataset.p_detections)
        lps_split = extract_tubes_from_video(
                                clip,
                                # gt=gt
                                )
        print('Num of tubes: ', len(lps_split))
        # le = loc_error_tube_gt(lps_split[0], gt)
        # print('localization error: ', le)

        # for lp in lps_split:
        #     iou_tube_gt(lp, gt)


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
    
    
    m = MakeUCFCrime2LocalClips(
        root='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips',
        # root_normal='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/frames',
        path_annotations='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CrimeViolence2LocalDATASET/Txt annotations-longVideos',
        path_person_detections='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/ucfcrime2local',
        abnormal=True)
    paths, labels, annotations, annotations_p_detections, num_frames = m()
    print('paths: ', len(paths))
    print('labels: ', len(labels))
    print('annotations: ', len(annotations))
    print('annotations_p_detections: ', len(annotations_p_detections))
    print('num_frames: ', len(num_frames))
    i=24
    print(paths[i])
    print(labels[i])
    print(annotations[i])
    print(annotations_p_detections[i])
    print(num_frames[i])
    
    # for idx in range(len(paths)):
    #     if config['video'] in paths[idx].split('/'):
    #         anns = m.ground_truth_boxes(paths[idx],annotations[idx])
    #         m.plot(paths[idx], anns, lps)
    #         break
    # # anns = m.ground_truth_boxes(paths[idx],annotations[idx])
    # # m.plot(paths[idx], anns)

    from VioNet.customdatasets.ucfcrime2local_dataset import UCFCrime2LocalVideoDataset
    num_tubes = []
    i = 0
    threshold = 0.2
    tp = 0
    fp = 0
    y_true = []
    pred_scores = []
    for path, label, annotation, annotation_p_detections, n_frames in zip(paths, labels, annotations, annotations_p_detections, num_frames):
        # if i > 0 :
        #     break
        print('{}--video:{}, num_frames: {}'.format(i+1, path, n_frames))
        # print('----annotation:{}, p_detec: {}, {}'.format(annotation, annotation_p_detections, type(annotation_p_detections)))
        video_dataset = UCFCrime2LocalVideoDataset(
            path=path,
            sp_annotation=annotation,
            transform=transforms.ToTensor(),
            clip_len=n_frames,
            clip_temporal_stride=5
        )
        person_detections = JSON_2_videoDetections(annotation_p_detections)
        TUBE_BUILD_CONFIG['dataset_root'] = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips'
        TUBE_BUILD_CONFIG['person_detections'] = person_detections
        for clip, frames, gt in video_dataset:
            lps_split = extract_tubes_from_video(
                                    clip,
                                    # gt=gt
                                    )
            num_tubes.append(len(lps_split))
            # le = loc_error_tube_gt(lps_split[0],gt, threshold=0.5)
            # print('localization_error: ', le)
            stmp_iou = st_iou(lps_split[0], gt)
            # print('stmp_iou: ', stmp_iou)
            y_true.append('positive')
            pred_scores.append(stmp_iou)
            if stmp_iou >= threshold:
                tp += 1
            else:
                fp += 1
        i+=1
    
    # avg_num_tubes = sum(num_tubes) / len(num_tubes)
    # print('avg_num_tubes: ', avg_num_tubes)
    print('y_true: ', y_true, len(y_true))
    print('pred_scores: ', pred_scores, len(pred_scores))
    print('tp: ', tp)
    print('fp: ', fp)


    # pred_scores = [0.7, 0.3, 0.5, 0.6, 0.55, 0.9, 0.4, 0.2, 0.4, 0.3]
    # y_true = ["positive", "negative", "negative", "positive", "positive", "positive", "negative", "positive", "negative", "positive"]
    # threshold = 0.5
    # y_pred = ['positive', 'negative', 'positive', 'positive', 'positive', 'positive', 'negative', 'negative', 'negative', 'negative']

    # Tp = 4
    # tn = 3
    # fp = 1
    # fn = 2


    # y_true = ["positive", "positive"]
    # pred_scores = [0.7, 0.3]
    # thresholds = np.arange(start=0.2, stop=0.7, step=0.05)

    thresholds = np.array([threshold])
    precisions, recalls = precision_recall_curve(y_true=y_true, 
                                                pred_scores=pred_scores,
                                                thresholds=thresholds)
    print('precisions: \n', precisions)                                            
    print('recalls: \n', recalls)      

    recall_11 = np.linspace(0, 1, 11)
    precisions_11 = []

    print('recall_11: \n', recall_11, recall_11.shape)
    for r in recall_11:
        if r <= recalls[0]:
            precisions_11.append(precisions[0])
        else:
            precisions_11.append(0)

    # matplotlib.pyplot.plot(recall_11, precisions_11, linewidth=4, color="red", zorder=0)
    # matplotlib.pyplot.xlabel("Recall", fontsize=12, fontweight='bold')
    # matplotlib.pyplot.ylabel("Precision", fontsize=12, fontweight='bold')
    # matplotlib.pyplot.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    # matplotlib.pyplot.show()

    # precisions = np.array(precisions)
    # recalls = np.array(recalls)

    AP = (1/11)*np.sum(precisions_11)
    print('AP: ', AP)

    # from numpy import trapz


    # # The y values.  A numpy array is used here,
    # # but a python list could also be used.
    # y = np.array(precisions)

    # # Compute the area using the composite trapezoidal rule.
    # area = trapz(y, dx=0.1)
    # print("area =", area)





    
    

    
            
    
    
        