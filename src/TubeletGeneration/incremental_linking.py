
from genericpath import isdir
import os
from sys import flags
import cv2
import numpy as np
import visual_utils

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
        # self.plot_wait = 1000
        # self.max_num_motion_boxes = 4
        
    
    def path_count(self, paths):
        return len(paths)
    
    def merge_close_detections(self, pred_boxes, flac, repetitions=5, only_merged=True):
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
        while r<repetitions:
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
        
    def get_temporal_window(self, t, windows):
        for w in windows:
            if t in w:
                return w

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
            split = self.video_detections[f]['split']
            video = self.video_detections[f]['video']
            frame = self.video_detections[f]['fname']
            img_path = os.path.join(self.dataset_root, split, video, frame)
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

    def __call__(self, frames, segmentor, plot=None, debug=False):
        live_paths = []
        dead_paths = []
        start_linking = False
        # MOTION_MAP = segmentor(frames)
        images_to_video = []
        plot_wait_1 = 1000#4000
        plot_wait_2 = 1000#6000
        # segmentor.plot(MOTION_MAP, wait=3000)
        # avg_image = segmentor.compute_mean_image(frames)
        windows = list(self.split_by_windows(frames, self.config['temporal_window']))
        current_window = None
        print('windows: ', windows)
        # if debug:
        #     save_folder = os.path.join(
        #         '/Users/davidchoqueluqueroman/Downloads/TubeGenerationExamples', 
        #         video_name)
        debug_folder = '/Users/davidchoqueluqueroman/Downloads/TubeGenerationExamples'
        for t in frames:
            #get current temporal window
            w = self.get_temporal_window(t, windows)
            img_paths, images = self.read_segment(w)
            
            #PLot and save results
            debug_motion_segm = debug
            video_name = img_paths[0].split('/')[-2]
            save_folder = os.path.join(debug_folder, video_name)
            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)

            save_folder_motion = os.path.join(
                save_folder,
                'motion_map')
            if not os.path.isdir(save_folder_motion):
                os.mkdir(save_folder_motion)
            #initialize segmentation
            if current_window == None:
                current_window = {
                    'map': segmentor(images, img_paths, debug_motion_segm, os.path.join(save_folder, str(w))),
                    'frames': w
                    }
            else:
                if not t in current_window['frames']:
                    current_window = {
                        'map': segmentor(images, img_paths, debug_motion_segm, os.path.join(save_folder, str(w))),
                        'frames': w
                    }
            #initialize tube building
            num_persons = self.video_detections[t]['pred_boxes'].shape[0]
            if num_persons == 0: #no persons detected in frame
                if debug:
                    print('\t*** NO PERSONS DETECTED at frame:{}'.format(t))
                    img = self.plot_frame(t, None, motion_map=None, plot_wait=plot_wait_1)
                    images_to_video.append(img)
                if not start_linking:
                    continue
                else:
                    lp_count = self.path_count(live_paths)
                    for lp in range(lp_count):
                        live_paths[lp]['lastfound'] += 1
            else:
                self.video_detections[t]['pred_boxes'][:, 4] = np.arange(1,self.video_detections[t]['pred_boxes'].shape[0]+1)
                person_detections = self.video_detections[t]['pred_boxes']

            if not start_linking:
                merge_pred_boxes = self.merge_close_detections(person_detections, only_merged=False, flac=FLAC_MERGED_1)
                # print('--frame {}, persons={}, merged boxes={}'.format(t, person_detections.shape, merge_pred_boxes.shape))
                # print('----persons:\n', self.video_detections[t]['pred_boxes'])
                # print('----merged boxes:\n', merge_pred_boxes)
                if debug:
                    img = self.plot_frame(t, merge_pred_boxes, motion_map=None, plot_wait=plot_wait_1)
                    images_to_video.append(img)
                # segmentor.plot(MOTION_MAP, bbox=merge_pred_boxes[0], wait=10000)
                # merge_pred_boxes = segmentor.filter_no_motion_boxes(merge_pred_boxes,
                #                                                     MOTION_MAP)
                # segmentor.plot_sub_motion_imgs(MOTION_MAP, wait=5000)
                # segmentor.plot(MOTION_MAP, lbbox=merge_pred_boxes, wait=1000)
                
                #init tubes
                num_persons = merge_pred_boxes.shape[0] #update number of bboxes in frame
                for b in range(num_persons):
                    if merge_pred_boxes[b,4] == -1:
                        # print('\t*** STARTED NEW TUBE at frame:{}'.format(t))
                        live_paths.append(
                            {
                                'frames_name': [self.video_detections[t]['fname']],
                                'boxes':[merge_pred_boxes[b,:]],
                                'len': 1, ##length of tube
                                'id': b, #id tube
                                'foundAt': [t],
                                'lastfound': 0, #diff between current frame and last frame in path
                                'score': merge_pred_boxes[b,4]
                            }
                        )
                        # print('live_paths[-1]: ', live_paths[-1]['boxes'][0].shape)
                        if debug:
                            img = self.plot_tube_frame(t,live_paths[-1], color='blue', plot_wait=plot_wait_2)
                    else:
                        # print('\t*** NO CLOSE PERSONS FOUND at frame:{}'.format(t))
                        if debug:
                            img = self.plot_frame(t, merged_boxes=merge_pred_boxes, motion_map=None, plot_wait=plot_wait_1)
                    if debug:
                        images_to_video.append(img)
                start_linking = True if len(live_paths)>0 else False
                continue
            
            if start_linking:
                person_detections = self.video_detections[t]['pred_boxes']
                merge_pred_boxes = []
                if len(person_detections) == 0:
                    # print('********** Skipped frame {} because no persons detected'.format(t))
                    if debug:
                        img = self.plot_frame(t,merged_boxes=None, motion_map=None, plot_wait=plot_wait_1)
                        images_to_video.append(img)
                    for lp in range(lp_count):
                        live_paths[lp]['lastfound'] += 1
                    continue

                # print('********** Started LINKING from frame {}'.format(t))
                merge_pred_boxes = self.merge_close_detections(person_detections, only_merged=False, flac=FLAC_MERGED_1)
                # print('--frame {}, persons={}, merged boxes={}'.format(t, person_detections.shape, merge_pred_boxes.shape))
                if debug:
                    self.plot_frame(t, merge_pred_boxes, motion_map=None, plot_wait=plot_wait_1)
                # merge_pred_boxes = segmentor.filter_no_motion_boxes(merge_pred_boxes,
                #                                                     MOTION_MAP)
                # print('********** Find Persons near tubes at frame {}'.format(t))
                #find persons near tubes
                for j, lp in enumerate(live_paths):
                    # print('#### tube {}'.format(j))
                    # print('####### persons={} shape {}'.format(merge_pred_boxes, merge_pred_boxes.shape))
                    final_box = self.merge_persons_to_tube(merge_pred_boxes, lp, thresh=0.3, flac=FLAC_MERGED_2)
                    # print('####### final_box={}'.format(final_box))
                    if final_box is not None:
                        lp['frames_name'].append(self.video_detections[t]['fname'])
                        lp['len'] += 1
                        lp['boxes'].append(final_box[0,:])
                        lp['foundAt'].append(t)
                        if debug:
                            img = self.plot_tube_frame(t,lp, color='green', plot_wait=plot_wait_2)
                            images_to_video.append(img)
                    else:
                        lp['lastfound'] += 1
                
                #start new paths
                for i in range(merge_pred_boxes.shape[0]):
                    if merge_pred_boxes[i,4] != FLAC_MERGED_2 and merge_pred_boxes[i,4] == FLAC_MERGED_1:
                        # print('\t*** STARTED NEW TUBE at frame:{}'.format(t))
                        live_paths.append(
                            {
                                'frames_name': [self.video_detections[t]['fname']],
                                'boxes':[merge_pred_boxes[i,:]],
                                'len': 1, ##length of tube
                                'id': len(live_paths), #id tube
                                'foundAt': [t],
                                'lastfound': 0, #diff between current frame and last frame in path
                                'score': merge_pred_boxes[i,4]
                            }
                        )
                        if debug:
                            img = self.plot_tube_frame(t,live_paths[-1], color='blue', plot_wait=plot_wait_2)
                            images_to_video.append(img)
                    else:
                        if debug:
                            img = self.plot_frame(t,merge_pred_boxes,motion_map=None, plot_wait=plot_wait_2)
                            images_to_video.append(img)

                    # for lp in live_paths:    
                        # # verify if there is a live path not detected in this frame but includes actual region
                        # is_dead_lp = lp['lastfound'] > self.jumpgap
                        # if not self.is_inside(merge_pred_boxes[i,:], lp['boxes'][-1]) and not is_dead_lp:
                        #     print('\t*** STARTED NEW TUBE at frame:{}'.format(t))
                        #     live_paths.append(
                        #         {
                        #             'frames_name': [self.video_detections[t]['fname']],
                        #             'boxes':[merge_pred_boxes[i,:]],
                        #             'len': 1, ##length of tube
                        #             'id': len(live_paths), #id tube
                        #             'foundAt': [t],
                        #             'lastfound': 0, #diff between current frame and last frame in path
                        #             'score': merge_pred_boxes[i,4]
                        #         }
                        #     )
                        #     img = self.plot_tube_frame(t,live_paths[-1], color='blue', plot_wait=plot_wait_2)
                        # else:
                        #     img = self.plot_frame(t,merge_pred_boxes, plot_wait=plot_wait_1)
                        # images_to_video.append(img)
                
                #terminate dead paths
                # live_paths, dead_paths = self.terminate_paths(live_paths, self.jumpgap)

                # num_boxes = merge_pred_boxes.shape[0] #update number of bboxes in frame
                # if len(live_paths)==0: #first frame
                #     for b in range(num_boxes):
                #         print('\t*** STARTED NEW TUBE at frame:{}'.format(t))
                #         live_paths.append(
                #             {
                #                 'frames_name': [self.video_detections[t]['fname']],
                #                 # 'boxes':[video_detections[t]['pred_boxes'][b,:]], ## Tube/list of bboxes
                #                 'boxes':[merge_pred_boxes[b,:]],
                #                 'len': 1, ##length of tube
                #                 'id': b, #id tube
                #                 'foundAt': [t],
                #                 'lastfound': 0, #diff between current frame and last frame in path
                #                 'score': merge_pred_boxes[b,4]
                #             }
                #         )
                # else:
                #     lp_count = self.path_count(live_paths)
                #     matrices = []
                #     covered_boxes = np.zeros(num_boxes)
                #     for lp in range(lp_count):
                #         linking_ious = self.iou_path_box(live_paths[lp], merge_pred_boxes[:,:4], self.iou_thresh)
                #         matrices.append(linking_ious)
                #     for lp in range(lp_count): #check over live tubes
                #         if live_paths[lp]['lastfound'] < self.jumpgap: #verify if path is possibly dead
                #             box_to_lp_score = matrices[lp]
                #             if np.sum(box_to_lp_score) > self.iou_thresh: #there is at least on box that match the tube
                #                 maxInd = np.argmax(box_to_lp_score)
                #                 max_score = np.max(box_to_lp_score)
                #                 live_paths[lp]['frames_name'].append(self.video_detections[t]['fname'])
                #                 live_paths[lp]['len'] += 1
                #                 live_paths[lp]['boxes'].append(merge_pred_boxes[maxInd,:])
                #                 live_paths[lp]['foundAt'].append(t)
                #                 covered_boxes[maxInd] = 1
                #                 # live_paths[lp]['score']+=merge_pred_boxes[maxInd,4]
                #             else:
                #                 live_paths[lp]['lastfound'] += 1
                #     ## terminate dead paths
                #     live_paths, dead_paths = self.terminate_paths(live_paths, self.jumpgap)
                #     ## join dead paths
                #     for dp in range(len(dead_paths)):
                #         dead_paths[dp]['id']
                #         live_paths.append(dead_paths[dp])
                #     lp_count = self.path_count(live_paths)
                #     ##start new paths/tubes
                #     if np.sum(covered_boxes) < num_boxes:
                #         for b in range(num_boxes):
                #             if not covered_boxes.flatten()[b]:
                #                 # print('start other tube at:', self.video_detections[t]['fname'])
                #                 live_paths.append(
                #                     {
                #                         'frames_name': [self.video_detections[t]['fname']],
                #                         # 'boxes':[video_detections[t]['pred_boxes'][b,:]], ## Tube/list of bboxes
                #                         'boxes':[merge_pred_boxes[b,:]],
                #                         'len': 1, ##length of tube
                #                         'id': lp_count, #id tube
                #                         'foundAt': [t],
                #                         'lastfound': 0, #diff between current frame and last frame in path
                #                         'score': merge_pred_boxes[b,4]
                #                     }
                #                 )
                #                 lp_count += 1
        self.fill_gaps(live_paths)
        if debug:
            create_video(images_to_video, debug_folder , 'tube_gen.avi', save_frames=True)
        # live_paths = sorted(live_paths, key = lambda i: i['score'], reverse=True)
        # print('live_paths before: ', len(live_paths))

        # for i in range(len(live_paths)):
        #     lp = live_paths[i]
        #     print('id:{}, foundAt: {}, len: {}, lastfound: {}'.format(lp['id'], lp['foundAt'], lp['len'], lp['lastfound']))

        
        # live_paths = self.remove_short_paths(live_paths,16)
        
        # if len(live_paths)==0:
        #     motion_boxes = MOTION_MAP['boxes_from_polygons']
        #     motion_boxes = [segmentor.contour_2_box(cnt) for cnt in motion_boxes]
        #     max_num_tubes = self.max_num_motion_boxes if len(motion_boxes)>self.max_num_motion_boxes else len(motion_boxes)
        #     for i in range(max_num_tubes):
        #         live_paths.append(
        #             {
        #                 'frames_name': [],
        #                 'boxes':[],
        #                 'len': 0, ##length of tube
        #                 'id': i, #id tube
        #                 'foundAt': [],
        #                 'lastfound': -1, #diff between current frame and last frame in path
        #                 'score': 0
        #             }
        #         )

        #     for j,t in enumerate(frames):
        #         for lp in range (len(live_paths)):
        #             live_paths[lp]['frames_name'].append(self.video_detections[t]['fname'])
        #             live_paths[lp]['len'] += 1
        #             live_paths[lp]['boxes'].append(motion_boxes[lp])
        #             live_paths[lp]['foundAt'].append(t)
        #             live_paths[lp]['score']= 0.5
        
        if plot:
            self.plot_tubes(frames, None, live_paths, plot['wait'])
        return live_paths
    
    def fill_gaps(self, live_paths):
        for i in range(len(live_paths)):
            lp = live_paths[i]
            full_path = list(range(lp['foundAt'][0],lp['foundAt'][-1]+1))
            diff = list(set(full_path) ^ set(lp['foundAt']))
            diff.sort()
            # print('diff:',diff)
            for d in diff:
                idx = d - lp['foundAt'][0]
                # print('idx:',idx)
                #avg box
                b1 = lp['boxes'][idx-1]
                b2 = lp['boxes'][idx]
                avg_box = list(np.mean(np.array([b1,b2]), axis=0))
                lp['boxes'].insert(idx, avg_box)
                lp['foundAt'].insert(idx, d)
                lp['frames_name'].insert(idx, 'frame{}.jpg'.format(d+1))
                lp['len'] = lp['len'] + 1
                lp['lastfound'] -= 1
                # lp['score'] +=  lp['boxes'][idx,4]


    



    def plot_frame(self, frame_t, merged_boxes=None, motion_map=None, plot_wait=3000):
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
        if merged_boxes is not None:
            image = visual_utils.draw_boxes(image,
                                            merged_boxes[:, :4],
                                            # scores=tube_scores,
                                            # ids=tube_ids,
                                            line_thick=1, 
                                            line_color='red')
        if motion_map:
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
        if motion_map:
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
        #plot
        cv2.namedWindow(frame_name,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(frame_name, (600,600))
        image = cv2.resize(image, (600,600))
        cv2.imshow(frame_name, image)
        key = cv2.waitKey(plot_wait)#pauses for 3 seconds before fetching next image
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows()
        return image

    def plot_tubes(self, frames, motion_map, live_paths, plot_wait):
        for t in frames:
            # split = self.video_detections[t]['split']
            # video = self.video_detections[t]['video']
            # frame = self.video_detections[t]['fname']
            # img_path = os.path.join(self.dataset_root, split, video, frame)
            # image = cv2.imread(img_path, cv2.IMREAD_COLOR)

            img_path, image, pred_boxes = self.load_frame(t)
            frame_name = img_path.split('/')[-1][:-4]

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
            lp = self.path_count(live_paths)
            box_tubes = []
            tube_ids = []
            tube_scores = []
            # print('====frame: ', frame, ', t: ', t)
            for l in range(lp):
                # print('frame number: {}, live_path {}, frames in lp: {}'.format(t, live_paths[l]['id'], 
                #                                                     live_paths[l]['foundAt']))
                # foundAt = True if t in live_paths[l]['foundAt'] else False
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
                image = visual_utils.draw_boxes(image,
                                                box_tubes[:, :4],
                                                # scores=tube_scores,
                                                ids=tube_ids,
                                                line_thick=1, 
                                                line_color='red')
            cv2.namedWindow('FRAME'+frame_name,cv2.WINDOW_NORMAL)
            cv2.resizeWindow('FRAME'+frame_name, (600,600))
            image = cv2.resize(image, (600,600))
            cv2.imshow('FRAME'+frame_name, image)
            key = cv2.waitKey(plot_wait)#pauses for 3 seconds before fetching next image
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows() 