import numpy as np
import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def videoDetections_2_JSON(output_path: str, video_detections: list):
    """
    Args:
        output_path: Folder to save JSON's
        video_detections: list pf Dictionaries. Each dict has the format:
            frame_detection = {
                            "fname": os.path.split(img_path)[1], #frame name
                            "video": one_video, #video name
                            "split": sp, #train/val
                            "pred_boxes":  pred_boxes[:, :5], #numpy array
                            "tags": pred_tags_name #list of labels ['person', 'person', ...]
                        }
    """
    with open(output_path, 'w') as fout:
        json.dump(video_detections , fout, cls=NumpyArrayEncoder)

def JSON_2_videoDetections(json_file):
    with open(json_file, "r") as read_file:
        decodedArray = json.load(read_file)
        # print("decoded Array:", type(decodedArray), len(decodedArray))
        
        for f in decodedArray:
            f['pred_boxes'] = np.asarray(f['pred_boxes'])
        # print(decodedArray[0])
        return decodedArray

def tube_2_JSON(output_path: str, tube: list):
    """
    {
        'frames_name': [video_detections[t]['fname']], #list of
        'boxes':[merge_pred_boxes[b,:]], #list of numpy arrays
        'len': 1, #length of tube
        'id': b, #id tube
        'foundAt': [t], #list of list, each list are the frames in each tube
        'lastfound': 0 #diff between current frame and last frame in path
    }
    """
    with open(output_path, 'w') as fout:
        json.dump(tube , fout, cls=NumpyArrayEncoder)

def JSON_2_tube(json_file):
    """
    """
    with open(json_file, "r") as read_file:
        decodedArray = json.load(read_file)
        # print("decoded Array:", type(decodedArray), len(decodedArray))
        
        for f in decodedArray:
            for i, box in  enumerate(f['boxes']):
                f['boxes'][i] = np.asarray(f['boxes'][i])
        # print(decodedArray[0])
        return decodedArray


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
