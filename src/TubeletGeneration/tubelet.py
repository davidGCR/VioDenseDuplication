import os
import json
import numpy as np
import cv2
import visual_utils
from SORT import Sort

def JSON_2_videoDetections(json_file):
    """
    Load Spatial detections from  a JSON file.
    Return a List with length frames
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


def merge_close_detections(pred_boxes, score_thr=0.5, iou_thr=0.4):
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
        # merged_indices = list(set(merged_indices))
        # nomerged_indices = list(set(nomerged_indices))
        # nomerged_indices = [i for i in nomerged_indices if i not in merged_indices]
        # nomerged_indices.sort()
        # for i in nomerged_indices:
        #     merge_pred_boxes.append(pred_boxes[i,:])
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

def tracking(decodedArray, vname, dataset_path="/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames"):
    mot_tracker = Sort()
    start_tracking = False
    tracked_objects = None

    size = (224,224)
    result = cv2.VideoWriter('/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/videos_/{}.avi'.format(vname), 
                         cv2.VideoWriter_fourcc('M','J','P','G'),
                         10, size)


    for item in decodedArray: #frame by frame
        
        img_path = os.path.join(dataset_path, item['split'], item['video'], item['fname'])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        # print("image shape:", image.shape)

        pred_boxes = item["pred_boxes"]
        pred_tags_name = item['tags']

        if not start_tracking:
            merge_pred_boxes = merge_close_detections(pred_boxes)
            start_tracking = True if len(merge_pred_boxes)>0 else False
        # pred_boxes = merge_pred_boxes
        # print("pred_boxes: ", pred_boxes.shape)
        
        # if merge_pred_boxes is not None:
            # print("merge_pred_boxes: ", merge_pred_boxes.shape)
        
        if pred_boxes.shape[0] != 0:
            image = visual_utils.draw_boxes(image,
                                            pred_boxes[:, :4],
                                            # scores=pred_boxes[:, 4],
                                            # tags=pred_tags_name,
                                            line_thick=1, 
                                            line_color='white')
        if start_tracking:
            if tracked_objects is not None:
                pred_boxes = np.empty((0,5)) if pred_boxes.shape[0] == 0 else pred_boxes
                tracked_objects = mot_tracker.update(pred_boxes)
            else:
                merge_pred_boxes = np.stack(merge_pred_boxes)
                tracked_objects = mot_tracker.update(merge_pred_boxes)       
            
            image = visual_utils.draw_boxes(image,
                                            tracked_objects[:, :4],
                                            ids=tracked_objects[:, 4],
                                            # tags=["per-"]*tracked_objects.shape[0],
                                            line_thick=2, 
                                            line_color='green')

        result.write(image) # save video
        name = img_path.split('/')[-1].split('.')[-2]
        cv2.imshow(name, image)
        key = cv2.waitKey(10)#pauses for 3 seconds before fetching next image
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows()
    
    result.release()
    # cv2.destroyAllWindows()


if __name__=="__main__":
    vname = "_q5Nwh4Z6ao_0"
    decodedArray = JSON_2_videoDetections("/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/RWF-2000/train/Fight/{}.json".format(vname))
    # print("decodedArray: ", type(decodedArray), len(decodedArray))

    # pred_boxes = decodedArray[0]['pred_boxes']
    # print("pred_boxes: ", type(pred_boxes), pred_boxes.shape)

    # plot_image_detections(decodedArray, "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames")
    tracking(decodedArray, vname=vname)
