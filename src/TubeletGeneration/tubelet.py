import os
import json
import numpy as np
import cv2
import visual_utils
from SORT import Sort
from tube_utils import tube_2_JSON

def JSON_2_videoDetections(json_file):
    """
    Load Spatial detections from  a JSON file.
    Return a List with length frames. 
    An element in the list contain a dict with the format:
    {
        'fname': 'frame1.jpg',
        'video': '0_DzLlklZa0_3',
        'split': 'train/Fight',
        'pred_boxes': array[],
        'tags': list(['person', ...])
    }
    """
    with open(json_file, "r") as read_file:
        decodedArray = json.load(read_file)
        # print("decoded Array:", type(decodedArray), len(decodedArray))
        
        for f in decodedArray:
            f['pred_boxes'] = np.asarray(f['pred_boxes'])
        # print(decodedArray[0])
        return decodedArray



def bbox_iou_numpy(box1, box2):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def merge_bboxes(bbox1, bbox2):
    # print("bbox1:", bbox1.shape, bbox1[0])
    # print("bbox2:", bbox2.shape, bbox2[0])

    x1 = min(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = max(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    s = max(bbox1[4], bbox2[4])
    return np.array([x1, y1, x2, y2, s])


def merge_close_detections(pred_boxes, score_thr=0.5, iou_thr=0.4, only_merged=True):
    """
    Merge bounding boxes in a frame
    """
    merge_pred_boxes = []
    if pred_boxes.shape[0] > 1:
        iou_matrix = bbox_iou_numpy(pred_boxes, pred_boxes)
        # print("iou_matrix: \n", iou_matrix, iou_matrix.shape)
        il2 = np.tril_indices(iou_matrix.shape[0],-1)
        merged_indices = []
        nomerged_indices = []
        for i,j in zip(il2[0],il2[1]):
            # print("i:{}, j:{}".format(i,j))
            if iou_matrix[i,j] >= iou_thr:
                # print("to merge:", pred_boxes[i,:], pred_boxes[j, :])
                m = merge_bboxes(pred_boxes[i,:], pred_boxes[j, :])
                merge_pred_boxes.append(m)
                merged_indices.append(i)
                merged_indices.append(j)
            else:
                nomerged_indices.append(i)
                nomerged_indices.append(j)

        ##### MERGED_BOXES + NO_MERGED_BOXES
        if not only_merged:
            merged_indices = list(set(merged_indices))
            nomerged_indices = list(set(nomerged_indices))
            nomerged_indices = [i for i in nomerged_indices if i not in merged_indices]
            nomerged_indices.sort()
            for i in nomerged_indices:
                merge_pred_boxes.append(pred_boxes[i,:])
            # merge_pred_boxes = np.stack(merge_pred_boxes)
        # assert merge_pred_boxes.shape[0] <= pred_boxes.shape[0], "Merge error!!! merged = {}/ raw = {}".format(merge_pred_boxes.shape[0], pred_boxes.shape[0])
    # else:
    #     merge_pred_boxes = pred_boxes
    return merge_pred_boxes

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

def tracking(decodedArray, 
            vname,
            dataset_frames_path="/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames", 
            video_out_path = '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/videos_',
            plot=True):

    mot_tracker = Sort(iou_threshold=0.1)
    start_tracking = False
    tracked_objects = None

    if plot:
        size = (224,224)
        result = cv2.VideoWriter('{}/{}.avi'.format(video_out_path,vname), 
                            cv2.VideoWriter_fourcc('M','J','P','G'),
                            10, size)


    for i, item in enumerate(decodedArray): #frame by frame
        
        if plot:
            img_path = os.path.join(dataset_frames_path, item['split'], item['video'], item['fname'])
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        # print("image shape:", image.shape)

        pred_boxes = item["pred_boxes"]
        pred_tags_name = item['tags']

        if not start_tracking:
            merge_pred_boxes = merge_close_detections(pred_boxes, only_merged=True)
            start_tracking = True if len(merge_pred_boxes)>0 else False
        # pred_boxes = merge_pred_boxes
        # print("pred_boxes: ", pred_boxes.shape)
        
        # if merge_pred_boxes is not None:
            # print("merge_pred_boxes: ", merge_pred_boxes.shape)
        
        if pred_boxes.shape[0] != 0 and plot:
            image = visual_utils.draw_boxes(image,
                                            pred_boxes[:, :4],
                                            # scores=pred_boxes[:, 4],
                                            # tags=pred_tags_name,
                                            line_thick=1, 
                                            line_color='white')
        if start_tracking:
            if tracked_objects is not None: 
                if pred_boxes.shape[0] == 0:
                    pred_boxes = np.empty((0,5))
                    print('tracked_objects(no persons in frame):frame {}'.format(i+1),tracked_objects.shape)
                else:
                    pred_boxes = merge_close_detections(pred_boxes, only_merged=False) #merge close bboxes in every frame
                    pred_boxes = np.stack(pred_boxes)
                    tracked_objects = mot_tracker.update(pred_boxes)
                    print('tracked_objects:frame {}'.format(i+1),tracked_objects.shape, ' ids: ', tracked_objects[:,4])
            else:
                merge_pred_boxes = np.stack(merge_pred_boxes)
                tracked_objects = mot_tracker.update(merge_pred_boxes)
                print('--tracked_objects:frame {}'.format(i+1),tracked_objects.shape, ' ids: ', tracked_objects[:,4])      
            if plot:
                image = visual_utils.draw_boxes(image,
                                                tracked_objects[:, :4],
                                                ids=tracked_objects[:, 4],
                                                # tags=["per-"]*tracked_objects.shape[0],
                                                line_thick=2, 
                                                line_color='green')

        if plot:
            result.write(image) # save video
            name = img_path.split('/')[-1].split('.')[-2]
            cv2.imshow(name, image)
            key = cv2.waitKey(10)#pauses for 3 seconds before fetching next image
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows()
    if plot:
        result.release()
        # cv2.destroyAllWindows()
    
    return tracked_objects

def extract_tubes_from_dataset(dataset_frames_path, persons_frame_detections_path):
    videos_list = os.listdir(dataset_frames_path)
    for video in videos_list:
        decodedArray = JSON_2_videoDetections(os.path.join(persons_frame_detections_path, video+'.json'))
        tracking(decodedArray,
                vname=video,
                dataset_frames_path='/media/david/datos/Violence DATA/RWF-2000/frames',
                video_out_path='/media/david/datos/PAPERS-SOURCE_CODE/VioDenseDuplication/videos_')

################################### NEW LINKING ALG ################################
def iou_path_box(path, frame_boxes, iou_thresh):
    """
    Compute iou betwen a tube/path and bboxes
    
    Args:
        path: A path dict representing one tube.
        frame_boxes: A numpy array (N,5), N bboxes in frame.
        iou_thresh: The threshold between tube and bounding boxes
    
    Return:
        iou_matrix: A matrix 1xN with iou's
    """
    last_box_of_path = path['boxes'][path['len']-1].reshape(1,-1)

    # print('iou_path_box args:', last_box_of_path.shape, frame_boxes.shape)

    iou_matrix = bbox_iou_numpy(last_box_of_path, frame_boxes)
    return iou_matrix

def path_count(paths):
    return len(paths)

def terminate_paths(live_paths, jumpgap):
    dead_paths = []
    final_live_paths = []
    # lp_c = 0
    # dp_c = 0
    for lp in range(path_count(live_paths)):
        if live_paths[lp]['lastfound'] > jumpgap: #dead paths
            # live_paths[lp]['id'] = lp_c
            dead_paths.append(live_paths[lp])
            # lp_c += 1
        else:
            # live_paths[lp]['id'] = dp_c
            final_live_paths.append(live_paths[lp])
            # dp_c += 1
    
    return final_live_paths, dead_paths

def incremental_linking(start_frame, video_detections, iou_thresh, jumpgap, plot):
    num_frames = len(video_detections)
    live_paths = []
    dead_paths = []
    start_linking = False
    for t in range(start_frame, num_frames):
        num_boxes = video_detections[t]['pred_boxes'].shape[0]
        ##look for close persons
        if num_boxes == 0: #no persons detected in frame
            if not start_linking:
                continue
            else:
                lp_count = path_count(live_paths)
                for lp in range(lp_count):
                    live_paths[lp]['lastfound'] += 1

        elif not start_linking: #there are detections
            merge_pred_boxes = merge_close_detections(video_detections[t]['pred_boxes'], only_merged=True)
            start_linking = True if len(merge_pred_boxes)>0 else False
            # if start_linking:
                # print('Start tracking from {}'.format(t))


        if start_linking:
            if num_boxes == 0: #no persons detected in frame
                lp_count = path_count(live_paths)
                for lp in range(lp_count):
                    live_paths[lp]['lastfound'] += 1
                continue

            merge_pred_boxes = merge_close_detections(video_detections[t]['pred_boxes'], only_merged=False) if num_boxes>=2 else video_detections[t]['pred_boxes']#Join very close persons
            # print("video_detections at {}={}".format(t, video_detections[t]['pred_boxes'].shape[0]))
            # print("merge_pred_boxes at {}={}".format(t, len(merge_pred_boxes)))
            merge_pred_boxes = np.stack(merge_pred_boxes) if isinstance(merge_pred_boxes, list) else merge_pred_boxes #listo to numpy
            num_boxes = merge_pred_boxes.shape[0] #update number of bboxes in frame

            if len(live_paths)==0: #first frame
                for b in range(num_boxes):
                    live_paths.append(
                        {
                            'frames_name': [video_detections[t]['fname']],
                            # 'boxes':[video_detections[t]['pred_boxes'][b,:]], ## Tube/list of bboxes
                            'boxes':[merge_pred_boxes[b,:]],
                            'len': 1, ##length of tube
                            'id': b, #id tube
                            'foundAt': [t],
                            'lastfound': 0 #diff between current frame and last frame in path
                        }
                    )
            else:
                lp_count = path_count(live_paths)
                # print('live-paths:',lp_count)
                matrices = []
                covered_boxes = np.zeros(num_boxes)
                for lp in range(lp_count):
                    # linking_ious = iou_path_box(live_paths[lp], video_detections[t]['pred_boxes'][:,:4], iou_thresh)
                    linking_ious = iou_path_box(live_paths[lp], merge_pred_boxes[:,:4], iou_thresh)

                    # best_ios = np.argwhere(linking_ious>iou_thresh)[:,1]
                    # print("linking_ious: ", linking_ious, linking_ious.shape, best_ios.shape, 'argmax:',)
                    # print("linking_ious: ", linking_ious, linking_ious.shape, 'max indx:', np.argmax(linking_ious), 'max score:', np.max(linking_ious), np.sum(linking_ious))
                    matrices.append(linking_ious)
                
                # dead_count = 0        
                for lp in range(lp_count): #check over live tubes
                    if live_paths[lp]['lastfound'] < jumpgap: #verify if path is possibly dead
                        box_to_lp_score = matrices[lp]
                        if np.sum(box_to_lp_score) > iou_thresh: #there is at least on box that match the tube
                            maxInd = np.argmax(box_to_lp_score)
                            max_score = np.max(box_to_lp_score)
                            live_paths[lp]['frames_name'].append(video_detections[t]['fname'])
                            live_paths[lp]['len'] += 1
                            # live_paths[lp]['boxes'].append(video_detections[t]['pred_boxes'][maxInd,:])
                            live_paths[lp]['boxes'].append(merge_pred_boxes[maxInd,:])
                            live_paths[lp]['foundAt'].append(t)
                            covered_boxes[maxInd] = 1
                        else:
                            live_paths[lp]['lastfound'] += 1
                        #     dead_count += 1
                
                ## terminate dead paths
                live_paths, dead_paths = terminate_paths(live_paths, jumpgap)

                ## join dead paths
                for dp in range(len(dead_paths)):
                    dead_paths[dp]['id']
                    live_paths.append(dead_paths[dp])

                lp_count = path_count(live_paths)

                ##start new paths/tubes
                if np.sum(covered_boxes) < num_boxes:
                    for b in range(num_boxes):
                        if not covered_boxes.flatten()[b]:
                            live_paths.append(
                                {
                                    'frames_name': [video_detections[t]['fname']],
                                    # 'boxes':[video_detections[t]['pred_boxes'][b,:]], ## Tube/list of bboxes
                                    'boxes':[merge_pred_boxes[b,:]],
                                    'len': 1, ##length of tube
                                    'id': lp_count, #id tube
                                    'foundAt': [t],
                                    'lastfound': 0 #diff between current frame and last frame in path
                                }
                            )
                            lp_count += 1
                
                

            # print('Final paths at {}={}'.format(t,len(live_paths)))
            # print('Paths ---live_paths[lp][foundAt]=', [lp['foundAt'] for lp in live_paths])
            

        if plot is not None:
            dataset_frames_path = plot['dataset_root']
            split = video_detections[t]['split']
            video = video_detections[t]['video']
            frame = video_detections[t]['fname']
            img_path = os.path.join(dataset_frames_path, split, video, frame)
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            pred_boxes = video_detections[t]['pred_boxes'] #real bbox
            if pred_boxes.shape[0] != 0 and plot:
                image = visual_utils.draw_boxes(image,
                                                pred_boxes[:, :4],
                                                # scores=pred_boxes[:, 4],
                                                # tags=pred_tags_name,
                                                line_thick=1, 
                                                line_color='white')
            lp = path_count(live_paths)
            box_tubes = []
            tube_ids = []
            for l in range(lp):
                # print('---live_paths[lp][foundAt]:', lp)
                foundAt = True if t in live_paths[l]['foundAt'] else False
                if foundAt:
                    bbox = live_paths[l]['boxes'][-1]
                    box_tubes.append(bbox)
                    tube_ids.append(live_paths[l]['id'])
            print('box_tubes:',len(box_tubes))
            print('tube_ids:',len(tube_ids),tube_ids)
            if len(box_tubes)>0:
                box_tubes = np.array(box_tubes)
                image = visual_utils.draw_boxes(image,
                                                box_tubes[:, :4],
                                                # scores=pred_boxes[:, 4],
                                                ids=tube_ids,
                                                line_thick=2, 
                                                line_color='orange')
            cv2.imshow('FRAME'+str(t+1), image)
            key = cv2.waitKey(plot['wait'])#pauses for 3 seconds before fetching next image
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows()            

        # print('==>{} Live paths at {} frame'.format(len(live_paths), t)) 

    live_paths = sorted(live_paths, key = lambda i: i['id'])
    return live_paths

def extract_tubes_from_dataset(dataset_persons_detections_path, folder_out):
    """
        Args:
            dataset_persons_detections_path: Path to folder containing the person detections in JSON format
    """
    videos_list = os.listdir(dataset_persons_detections_path)
    for i, video_folder in enumerate(videos_list):
        assert '.json' in video_folder, 'Unrecognized format!!!'
        print("Processing ({}/{}), pt: {}/{} ...".format(i+1,len(videos_list), dataset_persons_detections_path, video_folder))
        person_detections = JSON_2_videoDetections("{}/{}".format(dataset_persons_detections_path, video_folder))
        live_paths = incremental_linking(start_frame=0,
                        video_detections=person_detections,
                        iou_thresh=0.3,
                        jumpgap=3,
                        plot=None)
        if not os.path.isdir(folder_out):
            os.makedirs(folder_out)
        tube_2_JSON(output_path=os.path.join(folder_out, video_folder+'.json'), tube=live_paths)


if __name__=="__main__":
    # vname = "89UQqPuR4Q4_0"
    # decodedArray = JSON_2_videoDetections("/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/RWF-2000/train/Fight/{}.json".format(vname))
    # decodedArray = JSON_2_videoDetections("/media/david/datos/Violence DATA/PersonDetections/RWF-2000/train/Fight/{}.json".format(vname))
    # print("decodedArray: ", type(decodedArray), len(decodedArray), decodedArray[0])

    # pred_boxes = decodedArray[0]['pred_boxes']
    # print("pred_boxes: ", type(pred_boxes), pred_boxes.shape)

    # plot_image_detections(decodedArray, "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames")
    
    # tubes = tracking(decodedArray,
    #             vname=vname,
    #             dataset_frames_path='/media/david/datos/Violence DATA/RWF-2000/frames',
    #             video_out_path='/media/david/datos/PAPERS-SOURCE_CODE/VioDenseDuplication/videos_',
    #             plot=True)

    # print('tubes:', type(tubes), len(tubes))

    # live_paths = incremental_linking(start_frame=0,
    #                     video_detections=decodedArray,
    #                     iou_thresh=0.3,
    #                     jumpgap=3,
    #                     plot=None
    #                     # plot={
    #                     #   'dataset_root':  '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames',
    #                     #   'wait': 300
    #                     # }
    #                     )
    # tube_2_JSON(output_path=vname+'.json', tube=live_paths)
    # print('Paths ---live_paths[lp][frames_name]=', [lp['frames_name'] for lp in live_paths])
    # print('Paths ---live_paths[lp][frames_name]=', [(len(lp['boxes']), lp['len']) for lp in live_paths])
    # print('Live Paths Final:', len(live_paths))

    ##processing RWF-2000 dataset
    path_in = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/RWF-2000'
    path_out = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/Tubes/RWF-2000'
    splits = ['train/Fight', 'tran/NonFight', 'val/Fight', 'val/NonFight']
    for sp in splits:
        extract_tubes_from_dataset(dataset_persons_detections_path=os.path.join(path_in, sp), folder_out=os.path.join(path_out, sp))