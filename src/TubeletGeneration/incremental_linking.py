
from genericpath import isdir
import os
from sys import flags
import cv2
import numpy as np
from numpy.lib.function_base import append
# from VioNet.utils import colors
# from VioNet.utils import colors
import visual_utils
import re

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
    # print('box1: ', box1.shape, '--box2: ', box2.shape)
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

def merge_bboxes(bbox1, bbox2, flac):
    """
    return:
        array of shape (1,5)
    """

    x1 = min(bbox1[0], bbox2[0])
    y1 = min(bbox1[1], bbox2[1])
    x2 = max(bbox1[2], bbox2[2])
    y2 = max(bbox1[3], bbox2[3])
    # s = max(bbox1[4], bbox2[4])
    # s = bbox1[4] + bbox2[4]
    s = flac
    # print('joined score: ', bbox1[4], bbox2[4], s)
    return np.array([x1, y1, x2, y2, s]).reshape(1,-1)

def merge_bboxes_numpy(bboxes, flac):
    """
    input:  
        (N,5) array
    return:
        array of shape (1,5)
    """
    # print('bboxes',bboxes, bboxes.shape)
    # print('jmin: ', np.amin(bboxes, axis=0))
    # print('jmax: ', np.amax(bboxes, axis=0))

    x1 = np.amin(bboxes, axis=0)[0]
    y1 = np.amin(bboxes, axis=0)[1]
    x2 = np.amax(bboxes, axis=0)[2]
    y2 = np.amax(bboxes, axis=0)[3]
    # s = max(bbox1[4], bbox2[4])
    s = flac
    # print('joined score: ', bbox1[4], bbox2[4], s)
    return np.array([x1, y1, x2, y2, s]).reshape(1,-1)

def create_video(images, image_folder, video_name, save_frames=False):
    height, width, layers = images[0].shape
    video_name = os.path.join(image_folder, video_name)
    video = cv2.VideoWriter(video_name, 0, 1, (width, height))
    # Appending the images to the video one by one
    for i, image in enumerate(images): 
        if save_frames:
            cv2.imwrite(image_folder + '/'+str(i+1)+'.jpg', image)
        video.write(image) 
      
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()  # releasing the video generated

FLAC_MERGED_1 = -1
FLAC_MERGED_2 = -2
FLAC_MERGED_3 = -3

class IncrementalLinking:
    def __init__(
        self, 
        # video_detections, 
        config,
        # iou_thresh, 
        # jumpgap, 
        # dataset_root
        ):
        self.config = config
        self.video_detections = self.config['person_detections']
        # self.frames =frames
        # self.segmentor = segmentor
        
        self.iou_thresh = self.config['min_iou_close_persons']
        self.jumpgap = self.config['jumpgap']
        self.dataset_root = self.config['dataset_root']
        self.min_window_len = self.config['min_window_len']
        self.train_mode = self.config['train_mode']
        self.img_size = self.config['img_size']
        # self.max_num_motion_boxes = 4
        
    
    def path_count(self, paths):
        return len(paths)
    
    def merge_close_detections(self, pred_boxes, flac, only_merged=True):
        """
        Merge bounding boxes in a frame
        input:
            pred_boxes: (N, 5) array, N bounding boxes
        return:
            merge_pred_boxes: (M, 5) array, where M <= N
        """
        input_boxes = pred_boxes.copy()
        r=0
        merged_boxes = []
        while r<self.config['close_persons_rep']:
            merge_pred_boxes = []
            if input_boxes.shape[0] > 2:
                iou_matrix = bbox_iou_numpy(input_boxes, input_boxes)
                il2 = np.tril_indices(iou_matrix.shape[0],-1)
                # print("merge_close_detections iou_matrix: {} , tril: {}x{}".format(iou_matrix.shape, len(il2[0]),len(il2[1])))
                merged_indices = []
                nomerged_indices = []
                for i,j in zip(il2[0],il2[1]):
                    # print("box_i:{}, box_j:{}, iou={}".format(i,j, iou_matrix[i,j]))
                    if iou_matrix[i,j] >= self.iou_thresh:
                        # print("to merge:", pred_boxes[i,:], pred_boxes[j, :])
                        if i not in merged_indices and j not in merged_indices:
                            m = merge_bboxes(input_boxes[i,:], input_boxes[j, :], flac)
                            merge_pred_boxes.append(m)
                            merged_indices.append(i)
                            merged_indices.append(j)
                    # else:
                    #     nomerged_indices.append(i)
                    #     nomerged_indices.append(j)
                # print('merged={}/{} at repetiton={}'.format(merged_indices, len(merge_pred_boxes), r))
                ##### MERGED_BOXES + NO_MERGED_BOXES
                if not only_merged:
                    merged_indices = list(set(merged_indices))
                    for i in range(input_boxes.shape[0]):
                        if i not in merged_indices:
                            merge_pred_boxes.append(input_boxes[i,:].reshape(1,-1))

                    # nomerged_indices = list(set(nomerged_indices))
                #     nomerged_indices = [i for i in nomerged_indices if i not in merged_indices]
                #     nomerged_indices.sort()
                #     for i in nomerged_indices:
                #         merge_pred_boxes.append(input_boxes[i,:].reshape(1,-1))
                # print('merge_pred_boxes list:', len(merge_pred_boxes))
                # print('merged+no_merged={} at repetiton={}'.format(len(merge_pred_boxes), r))
                if len(merge_pred_boxes)==0:
                    break
                merge_pred_boxes = np.concatenate(merge_pred_boxes,axis=0)
                input_boxes = merge_pred_boxes
            else:
                merge_pred_boxes = input_boxes
                break
            r += 1
        return merge_pred_boxes
    
    def merge_persons_to_tube(self, persons, tube, flac, thresh=0.3):
        # for i in range(num_persons):
        # print('iou inputs: ', persons.shape, tube['boxes'][-1].shape)
        iou_matrix = bbox_iou_numpy(persons, tube['boxes'][-1].reshape(1,-1))
        # print('\tiou person x tube:\n', iou_matrix.shape)
        # print('\tpersons:', persons.shape, ', tube[boxes][-1]: ', tube['boxes'][-1].reshape(1,-1).shape)
        # il2 = np.tril_indices(iou_matrix.shape[0],-1)
        to_merge = []
        final_box = None
        for i in range(iou_matrix.shape[0]):
            for j in range(iou_matrix.shape[1]):
                # print("box_i:{}, box_j:{}, iou={}".format(i,j, iou_matrix[i,j]))
                if iou_matrix[i,j] >= thresh:
                    # print("to merge:", pred_boxes[i,:], pred_boxes[j, :])
                    persons[i, 4] = -2
                    to_merge.append(persons[i, :].reshape(1,-1))
                    # m = merge_bboxes(persons[i,:], persons[j, :])
                    # merge_pred_boxes.append(m)
        if len(to_merge)>0:
            to_merge = np.concatenate(to_merge, axis=0)
            final_box = merge_bboxes_numpy(to_merge, flac=flac)
        return final_box
        
    def merged_persons_motion(self, merged_detections, motion_regions, flac):
        iou_matrix = bbox_iou_numpy(merged_detections, motion_regions)
        # print('iou_matrix: \n', iou_matrix)
        for i in range(iou_matrix.shape[0]):
            for j in range(iou_matrix.shape[1]):
                if iou_matrix[i,j] >= 0.4:
                    # print('({},{})={}'.format(i,j, iou_matrix[i,j]))
                    if merged_detections[i,4] != FLAC_MERGED_2:
                        merged_detections[i,4]=flac
    
    def terminate_paths(self, live_paths, jumpgap):
        dead_paths = []
        final_live_paths = []
        # lp_c = 0
        # dp_c = 0
        for lp in range(self.path_count(live_paths)):
            if live_paths[lp]['lastfound'] > jumpgap: #dead paths
                # live_paths[lp]['id'] = lp_c
                dead_paths.append(live_paths[lp])
                # lp_c += 1
            else:
                # live_paths[lp]['id'] = dp_c
                final_live_paths.append(live_paths[lp])
                # dp_c += 1
        
        return final_live_paths, dead_paths
    
    def remove_short_paths(self, live_paths, min_len=5):
        final_live_paths = []
        for i in range(len(live_paths)):
            if len(live_paths[i]['foundAt']) > min_len: #dead paths
                # print('sspath: ', len(live_paths[i]['foundAt']),type(live_paths[i]))
                # live_paths[lp]['id'] = lp_c
                final_live_paths.append(live_paths[i])
                # lp_c += 1
        # print('removed---', type(final_live_paths), type(final_live_paths[0]))
        return final_live_paths

    def iou_path_box(self, path, frame_boxes, iou_thresh):
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

    def is_inside(self, box1, box2):
        """
        Verify if box1 is inside box2
        inputs:
            box1: (5,) array
            box2: (5,) array
        """
        tmp = box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]
        return tmp
    
    def split_by_windows(self, frames, window_len):
        for i in range(0, len(frames), window_len): 
            yield frames[i:i + window_len]
    
    def remove_short_windows(self, windows):
        c_final = []
        for c in windows:
            if len(c) >= self.min_window_len:
                c_final.append(c)
        return c_final
        
    def get_temporal_window(self, t, windows):
        for w in windows:
            if t in w:
                return w
        return None

    def read_segment(self, segment):
        """
        input:
            list of frames paths
        output: (img_paths, images)
            img_paths: list of image paths
            images: list of np.array images [(224,224,3), (224,224,3), ...]
        """
        img_paths = []
        images = []
        for f in segment:
            # print('f inm segment: ', f, len(self.video_detections))
            split = self.video_detections[f]['split']
            video = self.video_detections[f]['video']
            frame = self.video_detections[f]['fname']
            img_path = os.path.join(self.dataset_root, split, video, frame)
            assert os.path.isfile(img_path), print('File: {} does not exist!!!'.format({img_path}))
            img_paths.append(img_path)
            images.append(np.array(visual_utils.imread(img_path)))
        return img_paths, images
    
    def load_frame(self, frame_t):
        split = self.video_detections[frame_t]['split']
        video = self.video_detections[frame_t]['video']
        frame_name = self.video_detections[frame_t]['fname']
        img_path = os.path.join(self.dataset_root, split, video, frame_name)
        # image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = np.array(visual_utils.imread(img_path))

        person_boxes = self.video_detections[frame_t]['pred_boxes'] #real bbox

        return img_path, image, person_boxes
    
    def get_boxes_from_dict(self, motion_map, frame_t):
        boxes = []
        for fff in motion_map:
            if fff['frame'] == self.video_detections[frame_t]['fname']:
                boxes = fff['top_k_m_regions']
                boxes = [[b['x1'], b['y1'], b['x2'], b['y2'], b['id']] for b in boxes]
                boxes = np.array(boxes)
                break
        return boxes
    
    def is_dead(self, lp):
        return lp['lastfound'] > self.jumpgap

    def __call__(self, frames_indices, frames_names, segmentor, gt=None):
        live_paths = []
        dead_paths = []
        video_windows = []
        start_linking = False
        images_to_video = []
        plot = self.config['plot_config']['plot_tubes']
        plot_wait_1 = self.config['plot_config']['plot_wait_tubes']#4000
        plot_wait_2 = self.config['plot_config']['plot_wait_2']#6000
        save_results = self.config['plot_config']['save_results']
        debug_mode = self.config['plot_config']['debug_mode']
        windows = list(self.split_by_windows(frames_indices, self.config['temporal_window']))
        windows =  self.remove_short_windows(windows)
        current_window = None
        for t in frames_indices:
            #get current temporal window
            w = self.get_temporal_window(t, windows)
            if w == None:
                continue
            img_paths, images = self.read_segment(w)
            #initialize motion segmentation
            if current_window == None:
                motion_regions_map = segmentor(images, img_paths)
                current_window = {
                    'motion_regions_map': motion_regions_map,
                    'frames_numbers': w,
                    # 'tube': motion_tube
                    }
                video_windows.append(current_window)
                # print('first---', current_window['frames_numbers'])
            else:
                if not t in current_window['frames_numbers']:
                    motion_regions_map = segmentor(images, img_paths)
                    current_window = {
                        'motion_regions_map': motion_regions_map,
                        'frames_numbers': w,
                        # 'tube': motion_tube
                    }
                    video_windows.append(current_window)
            # print('fname: {}, frame_idx: {} , window: {}, real_frame: {}'.format(t, t, w, self.video_detections[t]['fname']))
            #initialize tube building
            num_persons = self.video_detections[t]['pred_boxes'].shape[0]
            if num_persons == 0: #no persons detected in frame
                if debug_mode:
                    print('\t*** NO PERSONS DETECTED at frame:{}'.format(t))
                    # print('current_window[motion_regions_map]: ', current_window['motion_regions_map'])
                    img = self.plot_frame(t, gt=gt, merged_boxes=None, motion_map=current_window['motion_regions_map'], plot_wait=plot_wait_1)
                    images_to_video.append(img)
                if start_linking:
                   for lp in range(len(live_paths)):
                        live_paths[lp]['lastfound'] += 1
                else:    
                    continue
            else:
                self.video_detections[t]['pred_boxes'][:, 4] = np.arange(1,self.video_detections[t]['pred_boxes'].shape[0]+1)
                person_detections = self.video_detections[t]['pred_boxes']
                if debug_mode:
                    img = self.plot_frame(t, gt=gt, merged_boxes=None, motion_map=current_window['motion_regions_map'], plot_wait=plot_wait_1)
                    images_to_video.append(img)

            if not start_linking:
                merge_pred_boxes = self.merge_close_detections(person_detections, only_merged=False, flac=FLAC_MERGED_1)
                # print('--frame {}, persons={}, merged boxes={}'.format(t, person_detections.shape, merge_pred_boxes.shape))
                # print('----persons:\n', self.video_detections[t]['pred_boxes'])
                # print('----merged boxes:\n', merge_pred_boxes)
                motion_regions_numpy = self.get_boxes_from_dict(current_window['motion_regions_map'],t)
                # print('current_window[motion_regions_map]: ', current_window['motion_regions_map']) 

                # print('motion_regions_numpy: ', motion_regions_numpy, motion_regions_numpy.shape) 
                # print('merge_pred_boxes: ', merge_pred_boxes, merge_pred_boxes.shape) 
                if motion_regions_numpy.shape[0] == 0:
                    continue
                self.merged_persons_motion(merge_pred_boxes, motion_regions_numpy, FLAC_MERGED_3)

                if debug_mode:
                    img = self.plot_frame(t, gt=gt, merged_boxes=merge_pred_boxes, motion_map=current_window['motion_regions_map'], plot_wait=plot_wait_1)
                    images_to_video.append(img)
                # segmentor.plot(MOTION_MAP, bbox=merge_pred_boxes[0], wait=10000)
                # merge_pred_boxes = segmentor.filter_no_motion_boxes(merge_pred_boxes,
                #                                                     MOTION_MAP)
                # segmentor.plot_sub_motion_imgs(MOTION_MAP, wait=5000)
                # segmentor.plot(MOTION_MAP, lbbox=merge_pred_boxes, wait=1000)
                
                #init tubes
                num_persons = merge_pred_boxes.shape[0] #update number of bboxes in frame
                lp_id = 0
                for b in range(num_persons):
                    # if merge_pred_boxes[b,4] == -1:
                    if merge_pred_boxes[b,4] == FLAC_MERGED_3:
                        
                        live_paths.append(
                            {
                                'frames_name': [self.video_detections[t]['fname']],
                                'boxes':[merge_pred_boxes[b,:]],
                                'len': 1, ##length of tube
                                'id': lp_id, #id tube
                                'foundAt': [t],
                                'lastfound': 0, #diff between current frame and last frame in path
                                'score': merge_pred_boxes[b,4],
                                'type': 'normal'
                            }
                        )
                        lp_id += 1
                        # print('live_paths[-1]: ', live_paths[-1]['boxes'][0].shape)
                        if debug_mode:
                            print('\t*** STARTED NEW TUBE at frame:{}'.format(t))
                            img = self.plot_tube_frame(t,live_paths[-1], motion_map=current_window['motion_regions_map'], color='blue', plot_wait=plot_wait_2)
                    else:
                        # print('\t*** NO CLOSE PERSONS FOUND at frame:{}'.format(t))
                        if debug_mode:
                            img = self.plot_frame(t, gt=gt, merged_boxes=merge_pred_boxes, motion_map=None, plot_wait=plot_wait_1)
                    if debug_mode:
                        images_to_video.append(img)
                start_linking = True if len(live_paths)>0 else False
                continue
            
            if start_linking:
                person_detections = self.video_detections[t]['pred_boxes']
                merge_pred_boxes = []
                if len(person_detections) == 0:
                    # print('********** Skipped frame {} because no persons detected'.format(t))
                    if debug_mode:
                        img = self.plot_frame(t, gt=gt, merged_boxes=None, motion_map=current_window['motion_regions_map'], plot_wait=plot_wait_1)
                        images_to_video.append(img)
                    for lp in range(len(live_paths)):
                        live_paths[lp]['lastfound'] += 1
                    continue

                # print('********** Started LINKING from frame {}'.format(t))
                merge_pred_boxes = self.merge_close_detections(person_detections, only_merged=False, flac=FLAC_MERGED_1)
                # print('--frame {}, persons={}, merged boxes={}'.format(t, person_detections.shape, merge_pred_boxes.shape))
                if debug_mode:
                    self.plot_frame(t, gt=gt, merged_boxes=merge_pred_boxes, motion_map=current_window['motion_regions_map'], plot_wait=plot_wait_1)
                # merge_pred_boxes = segmentor.filter_no_motion_boxes(merge_pred_boxes,
                #                                                     MOTION_MAP)
                # print('********** Find Persons near tubes at frame {}'.format(t))
                #find persons near tubes
                for j, lp in enumerate(live_paths):
                    # print('#### tube {}'.format(j))
                    if not self.is_dead(lp):
                        # print('####### persons={} shape {}'.format(merge_pred_boxes, merge_pred_boxes.shape))
                        final_box = self.merge_persons_to_tube(merge_pred_boxes, lp, thresh=0.3, flac=FLAC_MERGED_2)
                        # print('####### final_box={}'.format(final_box))
                        if final_box is not None:
                            lp['frames_name'].append(self.video_detections[t]['fname'])
                            lp['len'] += 1
                            lp['boxes'].append(final_box[0,:])
                            lp['foundAt'].append(t)
                            if debug_mode:
                                img = self.plot_tube_frame(t,lp, color='green', plot_wait=plot_wait_2)
                                images_to_video.append(img)
                        else:
                            lp['lastfound'] += 1
                
                #start new paths
                motion_regions_numpy = self.get_boxes_from_dict(current_window['motion_regions_map'],t)
                # print('motion_regions_numpy: ', type(motion_regions_numpy), motion_regions_numpy.shape)
                if motion_regions_numpy.shape[0] == 0:
                    continue
                self.merged_persons_motion(merge_pred_boxes, motion_regions_numpy, FLAC_MERGED_3)
                
                for i in range(merge_pred_boxes.shape[0]):
                    # if merge_pred_boxes[i,4] != FLAC_MERGED_2 and merge_pred_boxes[i,4] == FLAC_MERGED_1:
                    if merge_pred_boxes[i,4] != FLAC_MERGED_2 and merge_pred_boxes[i,4] != FLAC_MERGED_1 and merge_pred_boxes[i,4] == FLAC_MERGED_3:
                        live_paths.append(
                            {
                                'frames_name': [self.video_detections[t]['fname']],
                                'boxes':[merge_pred_boxes[i,:]],
                                'len': 1, ##length of tube
                                'id': len(live_paths), #id tube
                                'foundAt': [t],
                                'lastfound': 0, #diff between current frame and last frame in path
                                'score': merge_pred_boxes[i,4],
                                'type': 'normal'
                            }
                        )
                        if debug_mode:
                            img = self.plot_tube_frame(t,live_paths[-1], motion_map=current_window['motion_regions_map'], color='blue', plot_wait=plot_wait_2)
                            images_to_video.append(img)
                    else:
                        if debug_mode:
                            print('\t*** STARTED NEW TUBE at frame:{}'.format(t))
                            img = self.plot_frame(t, gt=gt, merged_boxes=merge_pred_boxes, plot_wait=plot_wait_2)
                            images_to_video.append(img)

        
        if save_results:
            create_video(images_to_video, self.config['plot_config']['save_folder_debug'] , 'tube_gen.avi', save_frames=True)
        
        if len(live_paths)==0 and self.train_mode:
            print('No tubes in video....using motion map')
            live_path_from_motion = ({
                    'frames_name': [],
                    'boxes':[],
                    'len': 0, ##length of tube
                    'id': '', #id tube
                    'foundAt': [],
                    'lastfound': 0, #diff between current frame and last frame in path
                    'score': 0,
                    'type': 'motion'
                })
            motion_in_video = True
            for vw in video_windows:#for each clip
                # window = vw['map']
                motion_regions_map = vw['motion_regions_map']
                # print('motion_regions_map: ', motion_regions_map, len(motion_regions_map))
                # print('motion_regions_map_middle: ', motion_regions_map[int(len(motion_regions_map)/2)])
                motion_regions_map = motion_regions_map[int(len(motion_regions_map)/2)]
                if len(motion_regions_map['top_k_m_regions']) == 0:
                    motion_in_video = False
                    continue
                else:
                    motion_in_video = True
                    middle_box = [i for i in motion_regions_map['top_k_m_regions'] if i['id']==0][0]
                    middle_frame = motion_regions_map['frame']
                frames_numbers = vw['frames_numbers']
                # middle_idx = int(len(motion_regions_map['m_regions'])/2)
                # middle_box = motion_regions_map['m_regions'][middle_idx]
                # middle_frame = motion_regions_map['frames'][middle_idx]
                box = np.array([middle_box['x1'], middle_box['y1'], middle_box['x2'], middle_box['y2'], middle_box['id']])
                # print('middle_frame: ', middle_frame)
                # print('box: ', box)
                live_path_from_motion['frames_name'].append(middle_frame)
                live_path_from_motion['boxes'].append(box),
                live_path_from_motion['len'] += 1
                live_path_from_motion['id'] = 8
                live_path_from_motion['foundAt'].append(frames_numbers[int(len(frames_numbers)/2)])
            # print('frames_name: ', live_path_from_motion['frames_name'])
            if motion_in_video:
                live_paths = [live_path_from_motion]
            else:
                live_paths = []
            if len(live_paths) == 0:
                print('\tNo Motion maps in video ....generating random tube')

                random_path = {
                    'frames_name': [],
                    'boxes':[],
                    'len': 0, ##length of tube
                    'id': '', #id tube
                    'foundAt': [],
                    'lastfound': 0, #diff between current frame and last frame in path
                    'score': 0,
                    'type': 'random'
                }

                w_min=100
                h_min=50
                xmin = np.random.randint(10, self.img_size[0]/2)
                ymin = np.random.randint(10, self.img_size[1]/2)
                xmax = np.random.randint(xmin + w_min, self.img_size[0])
                ymax = np.random.randint(ymin + h_min, self.img_size[0])
                # xmax = xmin + w_min
                # ymax = ymin + h_min
                random_box = [xmin, ymin, xmax, ymax, 0]
                random_box = np.array(random_box)
                # print('random_box: ', random_box)
                for t in frames_indices:
                    random_path['frames_name'].append(self.video_detections[t]['fname'])
                    random_path['boxes'].append(random_box),
                    random_path['len'] += 1
                    random_path['id'] = 0
                    random_path['foundAt'].append(t)
                live_paths = [random_path]


        
        self.fill_gaps(live_paths, frames_indices, frames_names)
        
        
        if plot:
            self.plot_tubes(frames_indices, None, live_paths, self.config['plot_config']['plot_wait_tubes'], gt, self.config['plot_config']['save_folder_final'])
        return live_paths

    def fill_gaps(self, live_paths, real_indices, real_frame_names):
        for i in range(len(live_paths)):
            if not  self.is_dead(live_paths[i]):
                # print('Filling live_path: {}:\n{}'.format(i+1, live_paths[i]))
                foundAt = live_paths[i]['foundAt']
                framesNames = live_paths[i]['frames_name']

                start_idx = foundAt[0]
                end_idx = foundAt[-1]

                real_segment = real_indices[start_idx:end_idx+1]
                missed_indices = []
                missed_frames = []
                indices_to_insert = []
                for k_idx, j in enumerate(real_segment):
                    if not j in foundAt:
                        missed_indices.append(j)
                        missed_frames.append(real_frame_names[j])
                        indices_to_insert.append(k_idx)
                
                # print('missed_indices: ', missed_indices)
                # print('missed_frames: ', missed_frames)
                # print('indices_to_insert: ', indices_to_insert)
                new_founAt = foundAt.copy()
                new_boxes = live_paths[i]['boxes'].copy()
                new_frames_name = live_paths[i]['frames_name'].copy()
                for ii, mi in zip(indices_to_insert, missed_indices):
                    new_founAt.insert(ii, mi)
                    new_frames_name.insert(ii, real_frame_names[mi])
                    # new_boxes.insert(ii, live_paths[i]['boxes'][ii-1]) #replace with previous box
                    new_boxes.insert(ii, new_boxes[ii-1]) #replace with previous box

                live_paths[i]['foundAt'] = new_founAt  
                live_paths[i]['frames_name'] = new_frames_name  
                live_paths[i]['boxes'] = new_boxes  
                live_paths[i]['len'] += len(missed_frames)
                
                assert new_founAt == real_segment, 'Filling lp error!!!'
                # print('path: {}, foundAt: {}, real_segment: {}, missed_indices: {}, indices_to_insert: {} = {}'.format(i+1, foundAt, real_segment, missed_indices, indices_to_insert, new_founAt))

    def plot_frame(
        self, 
        frame_t,
        gt=None,
        merged_boxes=None, 
        motion_map=None, 
        plot_wait=3000):
        img_path, image, pred_boxes = self.load_frame(frame_t)
        # print('img_path: ', img_path.split('/')[-1], ' -t: ', frame_t)
        frame_name = img_path.split('/')[-1][:-4]
        if pred_boxes.shape[0] != 0:
            image = visual_utils.draw_boxes(image,
                                            pred_boxes[:, :4],
                                            # scores=pred_boxes[:, 4],
                                            # tags=pred_tags_name,
                                            ids=pred_boxes[:, 4],
                                            line_thick=2, 
                                            line_color='white')
        # ground truth
        if gt is not None:
            temp = int(re.findall(r'\d+', frame_name)[0])
            frame_gt = list(filter(lambda fgt: int(fgt['frame']) == temp, gt))[0]
            cv2.rectangle(image,
                        (int(frame_gt['xmin']), int(frame_gt['ymin'])),
                        (int(frame_gt['xmax']), int(frame_gt['ymax'])),
                        visual_utils.color['aqua'],
                        1)
        #merged persons
        if merged_boxes is not None:
            image = visual_utils.draw_boxes(image,
                                            merged_boxes[:, :4],
                                            # scores=tube_scores,
                                            # ids=tube_ids,
                                            line_thick=1, 
                                            line_color='red')
        if motion_map is not None:
            # countours = motion_map['contours']
            i = 0
            # boxes = motion_map[frame_t]['top_k_m_regions']
            # assert motion_map[frame_t]['frame'] == self.video_detections[frame_t]['fname'], 'No match motion: {}-current frame: {}'.format(motion_map[frame_t]['frame'], self.video_detections[frame_t]['fname'])
            boxes = []
            for fff in motion_map:
                if fff['frame'] == self.video_detections[frame_t]['fname']:
                    boxes = fff['top_k_m_regions']
                    break
            for i in range(len(boxes)):
                bc = boxes[i]
                cv2.rectangle(
                            image, 
                            (int(bc['x1']), int(bc['y1'])), 
                            (int(bc['x2']), int(bc['y2'])), 
                            visual_utils.color['deep pink'], 
                            2)
                cv2.putText(
                    image, 
                    str(bc['id']), 
                    (int(bc['x1']),int(bc['y1']) - 7), 
                    cv2.FONT_ITALIC, 
                    0.5, 
                    visual_utils.color['deep pink'], 2)
        #plot
        cv2.namedWindow(frame_name,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(frame_name, (600,600))
        image = cv2.resize(image, (600,600))
        cv2.imshow(frame_name, image)
        key = cv2.waitKey(plot_wait)#pauses for 3 seconds before fetching next image
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows()
        
        return image
    
    def plot_tube_frame(self, frame_t, tube, color, motion_map=None, plot_wait=3000):
        # split = self.video_detections[frame_t]['split']
        # video = self.video_detections[frame_t]['video']
        # frame = self.video_detections[frame_t]['fname']
        # img_path = os.path.join(self.dataset_root, split, video, frame)
        # image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # #Persons
        # pred_boxes = self.video_detections[frame_t]['pred_boxes'] #real bbox

        img_path, image, pred_boxes = self.load_frame(frame_t)
        frame_name = img_path.split('/')[-1][:-4]
        
        if pred_boxes.shape[0] != 0:
            image = visual_utils.draw_boxes(image,
                                            pred_boxes[:, :4],
                                            # scores=pred_boxes[:, 4],
                                            # tags=pred_tags_name,
                                            ids=pred_boxes[:, 4],
                                            line_thick=2, 
                                            line_color='white')
        #merged persons
        image = visual_utils.draw_boxes(image,
                                        [tube['boxes'][-1]],
                                        # scores=tube_scores,
                                        ids=[tube['id']],
                                        line_thick=1, 
                                        line_color=color)
        if motion_map is not None:
            # countours = motion_map['contours']
            i = 0
            boxes = []
            for fff in motion_map:
                if fff['frame'] == self.video_detections[frame_t]['fname']:
                    boxes = fff['top_k_m_regions']
                    break
            for i in range(len(boxes)):
                bc = boxes[i]
                cv2.rectangle(
                            image, 
                            (int(bc['x1']), int(bc['y1'])), 
                            (int(bc['x2']), int(bc['y2'])), 
                            visual_utils.color['deep pink'], 
                            2)
                cv2.putText(
                    image, 
                    str(bc['id']), 
                    (int(bc['x1']),int(bc['y1']) - 7), 
                    cv2.FONT_ITALIC, 
                    0.5, 
                    visual_utils.color['deep pink'], 
                    2
                    )
        #plot
        cv2.namedWindow(frame_name,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(frame_name, (600,600))
        image = cv2.resize(image, (600,600))
        cv2.imshow(frame_name, image)
        key = cv2.waitKey(plot_wait)#pauses for 3 seconds before fetching next image
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows()
        return image

    def plot_tubes(self, frames, motion_map, live_paths, plot_wait, gt=None, save_folder=None):
        images_to_video = []
        colors = []
        # print('====frame: ', frame, ', t: ', t)
        for l in range(len(live_paths)):
            b_color = (
                    np.random.randint(0,255), 
                    np.random.randint(0,255), 
                    np.random.randint(0,255)
                    )
            colors.append(b_color)
        for t in frames:
            img_path, image, pred_boxes = self.load_frame(t)
            # frame_name = img_path.split('/')[-1][:-4]
            frame_name = img_path.split('/')[-1]
            # print('frame_name: ',frame_name)

            #ground truth
            if gt is not None:
                temp = int(re.findall(r'\d+', frame_name)[0])
                frame_gt = list(filter(lambda fgt: int(fgt['frame']) == temp, gt))[0]
                cv2.rectangle(image,
                            (int(frame_gt['xmin']), int(frame_gt['ymin'])),
                            (int(frame_gt['xmax']), int(frame_gt['ymax'])),
                            visual_utils.color['aqua'],
                            2)
            #draw motion blobs
            if motion_map is not None:
                countours = motion_map['contours']
                boxes_from_polygons = motion_map['boxes_from_polygons']
                polygons = motion_map['polygons']
                for i in range(len(countours)):
                    # cv2.drawContours(image, countours, i,(255,0,0), 2)
                    cv2.drawContours(image, [polygons[i]], 0, (147,20,255), 1)
                    cv2.rectangle(image,
                                (int(boxes_from_polygons[i][0]), int(boxes_from_polygons[i][1])),
                                (int(boxes_from_polygons[i][0]+boxes_from_polygons[i][2]), int(boxes_from_polygons[i][1]+boxes_from_polygons[i][3])),
                                (0,238,238),
                                1)
            
            #Persons
            pred_boxes = self.video_detections[t]['pred_boxes'] #real bbox
            if pred_boxes.shape[0] != 0:
                image = visual_utils.draw_boxes(image,
                                                pred_boxes[:, :4],
                                                # scores=pred_boxes[:, 4],
                                                # tags=pred_tags_name,
                                                line_thick=1, 
                                                line_color='white')
            
            # live_paths = sorted(live_paths, key = lambda i: i['score'], reverse=True)
            # live_paths = live_paths[0:4] if len(live_paths)>4 else live_paths
            # lp = self.path_count(live_paths)
            box_tubes = []
            tube_ids = []
            tube_scores = []
            

            for l in range(len(live_paths)):
                # print('frame number: {}, live_path {}, frames in lp: {}'.format(t, live_paths[l]['id'], 
                #                                                     live_paths[l]['foundAt']))
                # print(live_paths[l])
                foundAt = True if frame_name in live_paths[l]['frames_name'] else False
                if foundAt:
                    # bbox = live_paths[l]['boxes'][-1]
                    idx = live_paths[l]['frames_name'].index(frame_name)
                    bbox = live_paths[l]['boxes'][idx]
                    # print('foundAt box: ',bbox, live_paths[l]['len'])
                    # print('foundAt all box: ',bbox, live_paths[l]['boxes'])
                    box_tubes.append(bbox)
                    tube_ids.append(live_paths[l]['id'])
                    tube_scores.append(live_paths[l]['score'])
            
            
            # print('box_tubes:',len(box_tubes))
            # print('tube_ids:',len(tube_ids),tube_ids)
            # print('tube_scores:',len(tube_scores),tube_scores)
            # print('====================')
            
                
            if len(box_tubes)>0:
                box_tubes = np.array(box_tubes)
                # print('iamge shape: ', image.shape)
                image = visual_utils.draw_boxes(image,
                                                box_tubes[:, :4],
                                                # scores=tube_scores,
                                                ids=tube_ids,
                                                line_thick=2, 
                                                line_color=colors)
            images_to_video.append(image)
            cv2.namedWindow('FRAME'+frame_name,cv2.WINDOW_NORMAL)
            cv2.resizeWindow('FRAME'+frame_name, (600,600))
            image = cv2.resize(image, (600,600))
            cv2.imshow('FRAME'+frame_name, image)
            key = cv2.waitKey(plot_wait)#pauses for 3 seconds before fetching next image
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows() 
        if save_folder is not None:
            o_path = os.path.join(save_folder, 'final')
            if not os.path.isdir(o_path):
                os.mkdir(o_path)
            create_video(images_to_video, o_path , 'only_tubes_gen.avi', save_frames=True)