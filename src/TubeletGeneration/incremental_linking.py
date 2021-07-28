
import os
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
    
class IncrementalLinking:
    def __init__(self, video_detections, iou_thresh, jumpgap, dataset_root):
        self.video_detections = video_detections
        # self.frames =frames
        # self.segmentor = segmentor
        self.iou_thresh = iou_thresh
        self.jumpgap = jumpgap
        self.dataset_root = dataset_root
        # self.plot_wait = 1000
        self.max_num_motion_boxes = 4
        self.ratio_tr = 0.3
    
    def path_count(self, paths):
        return len(paths)
    
    def merge_close_detections(self, pred_boxes, score_thr=0.5, iou_thr=0.4, only_merged=True):
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

    def __call__(self, frames, segmentor, plot=None):
        live_paths = []
        dead_paths = []
        start_linking = False
        MOTION_MAP = segmentor(frames)
        for t in frames:
            num_boxes = self.video_detections[t]['pred_boxes'].shape[0]
            ##look for close persons
            if num_boxes == 0: #no persons detected in frame
                if not start_linking:
                    continue
                else:
                    lp_count = self.path_count(live_paths)
                    for lp in range(lp_count):
                        live_paths[lp]['lastfound'] += 1
            elif not start_linking: #there are detections
                merge_pred_boxes = self.merge_close_detections(self.video_detections[t]['pred_boxes'], only_merged=True)
                # segmentor.plot(motion_map, bbox=merge_pred_boxes[0], wait=10000)
                merge_pred_boxes = segmentor.filter_no_motion_boxes(merge_pred_boxes,
                                                                    MOTION_MAP,
                                                                    self.ratio_tr)
                
                # segmentor.plot_sub_motion_imgs(MOTION_MAP, wait=5000)
                # segmentor.plot(MOTION_MAP, lbbox=merge_pred_boxes, wait=1000)
                start_linking = True if len(merge_pred_boxes)>0 else False
            
            if start_linking:
                if num_boxes == 0: #no persons detected in frame
                    lp_count = self.path_count(live_paths)
                    for lp in range(lp_count):
                        live_paths[lp]['lastfound'] += 1
                    continue
                #Join very close persons
                merge_pred_boxes = self.merge_close_detections(self.video_detections[t]['pred_boxes'], only_merged=False) if num_boxes>=2 else self.video_detections[t]['pred_boxes']
                merge_pred_boxes = segmentor.filter_no_motion_boxes(merge_pred_boxes,
                                                                    MOTION_MAP,
                                                                    self.ratio_tr)
                
                
                # print("merge_pred_boxes at {}={}".format(t, len(merge_pred_boxes))) #len(f_merge_pred_boxes)))

                if len(merge_pred_boxes) == 0: #no persons detected in frame
                    lp_count = self.path_count(live_paths)
                    for lp in range(lp_count):
                        live_paths[lp]['lastfound'] += 1
                    continue

                num_boxes = len(merge_pred_boxes)
                merge_pred_boxes = np.stack(merge_pred_boxes) if isinstance(merge_pred_boxes, list) and len(merge_pred_boxes)>0 else merge_pred_boxes #listo to numpy
                # num_boxes = merge_pred_boxes.shape[0] #update number of bboxes in frame

                if len(live_paths)==0: #first frame
                    for b in range(num_boxes):
                        live_paths.append(
                            {
                                'frames_name': [self.video_detections[t]['fname']],
                                # 'boxes':[video_detections[t]['pred_boxes'][b,:]], ## Tube/list of bboxes
                                'boxes':[merge_pred_boxes[b,:]],
                                'len': 1, ##length of tube
                                'id': b, #id tube
                                'foundAt': [t],
                                'lastfound': 0 #diff between current frame and last frame in path
                            }
                        )
                else:
                    lp_count = self.path_count(live_paths)
                    matrices = []
                    covered_boxes = np.zeros(num_boxes)
                    for lp in range(lp_count):
                        linking_ious = self.iou_path_box(live_paths[lp], merge_pred_boxes[:,:4], self.iou_thresh)
                        matrices.append(linking_ious)
                    for lp in range(lp_count): #check over live tubes
                        if live_paths[lp]['lastfound'] < self.jumpgap: #verify if path is possibly dead
                            box_to_lp_score = matrices[lp]
                            if np.sum(box_to_lp_score) > self.iou_thresh: #there is at least on box that match the tube
                                maxInd = np.argmax(box_to_lp_score)
                                max_score = np.max(box_to_lp_score)
                                live_paths[lp]['frames_name'].append(self.video_detections[t]['fname'])
                                live_paths[lp]['len'] += 1
                                live_paths[lp]['boxes'].append(merge_pred_boxes[maxInd,:])
                                live_paths[lp]['foundAt'].append(t)
                                covered_boxes[maxInd] = 1
                            else:
                                live_paths[lp]['lastfound'] += 1
                    ## terminate dead paths
                    live_paths, dead_paths = self.terminate_paths(live_paths, self.jumpgap)
                    ## join dead paths
                    for dp in range(len(dead_paths)):
                        dead_paths[dp]['id']
                        live_paths.append(dead_paths[dp])
                    lp_count = self.path_count(live_paths)
                    ##start new paths/tubes
                    if np.sum(covered_boxes) < num_boxes:
                        for b in range(num_boxes):
                            if not covered_boxes.flatten()[b]:
                                live_paths.append(
                                    {
                                        'frames_name': [self.video_detections[t]['fname']],
                                        # 'boxes':[video_detections[t]['pred_boxes'][b,:]], ## Tube/list of bboxes
                                        'boxes':[merge_pred_boxes[b,:]],
                                        'len': 1, ##length of tube
                                        'id': lp_count, #id tube
                                        'foundAt': [t],
                                        'lastfound': 0 #diff between current frame and last frame in path
                                    }
                                )
                                lp_count += 1
        
        live_paths = sorted(live_paths, key = lambda i: i['id'])
        # print('live_paths before fill: ', len(live_paths))

        # for i in range(len(live_paths)):
        #     lp = live_paths[i]
        #     print('id:{}, foundAt: {}, len: {}, lastfound: {}'.format(lp['id'], lp['foundAt'], lp['len'], lp['lastfound']))

        self.fill_gaps(live_paths)
        
        if len(live_paths)==0:
            motion_boxes = MOTION_MAP['boxes_from_polygons']
            motion_boxes = [segmentor.contour_2_box(cnt) for cnt in motion_boxes]
            max_num_tubes = self.max_num_motion_boxes if len(motion_boxes)>self.max_num_motion_boxes else len(motion_boxes)
            for i in range(max_num_tubes):
                live_paths.append(
                    {
                        'frames_name': [],
                        'boxes':[],
                        'len': 0, ##length of tube
                        'id': i, #id tube
                        'foundAt': [],
                        'lastfound': -1 #diff between current frame and last frame in path
                    }
                )

            for j,t in enumerate(frames):
                for lp in range (len(live_paths)):
                    live_paths[lp]['frames_name'].append(self.video_detections[t]['fname'])
                    live_paths[lp]['len'] += 1
                    live_paths[lp]['boxes'].append(motion_boxes[lp])
                    live_paths[lp]['foundAt'].append(t)
        if plot:
            self.plot_tubes(frames, MOTION_MAP, live_paths, plot['wait'])
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

    def plot_tubes(self, frames, motion_map, live_paths, plot_wait):
        for t in frames:
            split = self.video_detections[t]['split']
            video = self.video_detections[t]['video']
            frame = self.video_detections[t]['fname']
            img_path = os.path.join(self.dataset_root, split, video, frame)
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            #draw motion blobs
            countours = motion_map['contours']
            boxes_from_polygons = motion_map['boxes_from_polygons']
            polygons = motion_map['polygons']
            for i in range(len(countours)):
                # cv2.drawContours(image, countours, i,(255,0,0), 2)
                cv2.drawContours(image, [polygons[i]], 0, (147,20,255), 2)
                cv2.rectangle(image,
                            (int(boxes_from_polygons[i][0]), int(boxes_from_polygons[i][1])),
                            (int(boxes_from_polygons[i][0]+boxes_from_polygons[i][2]), int(boxes_from_polygons[i][1]+boxes_from_polygons[i][3])),
                            (0,238,238),
                            2)

            pred_boxes = self.video_detections[t]['pred_boxes'] #real bbox
            if pred_boxes.shape[0] != 0:
                image = visual_utils.draw_boxes(image,
                                                pred_boxes[:, :4],
                                                # scores=pred_boxes[:, 4],
                                                # tags=pred_tags_name,
                                                line_thick=1, 
                                                line_color='white')
            lp = self.path_count(live_paths)
            box_tubes = []
            tube_ids = []
            for l in range(lp):
                print('frame number: {}, live_path {}, frames in lp: {}'.format(t, live_paths[l]['id'], 
                                                                    live_paths[l]['foundAt']))
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
            key = cv2.waitKey(plot_wait)#pauses for 3 seconds before fetching next image
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows() 