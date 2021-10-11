from operator import itemgetter
import numpy as np
import torch
import random
import numpy as np


from global_var import MIDDLE, EVENLY, DYN_IMAGE, RGB_FRAME

class TubeCrop(object):
    def __init__(self, 
                    tube_len=16,
                    central_frame=True, 
                    max_num_tubes=4, 
                    train=True,
                    input_type=RGB_FRAME,
                    sample_strategy=MIDDLE,#'middle', #'evenly'
                    random=True):
        """
        Args:
        """
        self.tube_len = tube_len
        self.central_frame = central_frame
        self.max_num_tubes = max_num_tubes
        self.train = train
        self.input_type = input_type
        self.random = random
        self.sample_strategy = sample_strategy

    def __call__(self, tubes: list, max_video_len: int):
        segments = []
        boxes = []
        if not self.random:
            tubes = sorted(tubes, key = lambda i: i['len'], reverse=True)
        
        for tube in tubes:
            if self.input_type==RGB_FRAME:
                
                frames_idxs = self.__centered_frames__(tube['foundAt'],max_video_len)
            else:
                frames_idxs = self.__centered_segments__()
            if len(frames_idxs) > 0:
                bbox = self.__central_bbox__(tube['boxes'], tube['id']+1)
                boxes.append(bbox)
                segments.append(frames_idxs)
                # print(
                    # 'tube[foundAt]: ', tube['foundAt'] , 
                #         '\tsample: ', frames_idxs,
                        # '\tbox: ',bbox)   
        idxs = range(len(boxes))
        if self.max_num_tubes != 0 and len(boxes) > self.max_num_tubes:
            if self.random:
                idxs = random.sample(range(len(boxes)), self.max_num_tubes)
                boxes = list(itemgetter(*idxs)(boxes))
                # print('random boxes: ', boxes)
                segments = list(itemgetter(*idxs)(segments))
                segments = [segments] if self.max_num_tubes==1 else segments
                # if self.train:
                #     idxs = random.sample(range(len(boxes)), self.max_num_tubes)
                #     boxes = list(itemgetter(*idxs)(boxes))
                #     segments = list(itemgetter(*idxs)(segments))
                # else:
                #     n = len(boxes)
                #     m = int(n/2)
                #     boxes = boxes[m-int(self.max_num_tubes/2) : m+int(self.max_num_tubes/2)]
                #     segments = segments[m-int(self.max_num_tubes/2) : m+int(self.max_num_tubes/2)]
            else:
                boxes = boxes[0:self.max_num_tubes]
                segments = segments[0:self.max_num_tubes]
        
        # if len(boxes)==1:
        #     print('boxes==1: ', boxes, boxes[0].size())
        #     boxes[0] = torch.unsqueeze(boxes[0], 0)
        #     print('boxes unsqueeze: ', boxes, boxes[0].size())
        for id,box in enumerate(boxes):
            boxes[id][0,0] = id
        # print('boxes ids: ', boxes)
        return boxes, segments, idxs
    
    def __centered_frames__(self, tube_frames_idxs: list, max_video_len: int):
        if len(tube_frames_idxs) == self.tube_len: 
            return tube_frames_idxs
        if len(tube_frames_idxs) > self.tube_len:
            if self.sample_strategy == MIDDLE:
                n = len(tube_frames_idxs)
                m = int(n/2)
                arr = np.array(tube_frames_idxs)
                centered_array = arr[m-int(self.tube_len/2) : m+int(self.tube_len/2)]
            elif self.sample_strategy == EVENLY:
                min_frame = tube_frames_idxs[0]
                tube_frames_idxs = np.linspace(min_frame, max_video_len, self.tube_len).astype(int)
                tube_frames_idxs = tube_frames_idxs.tolist()

            return centered_array.tolist()
        if len(tube_frames_idxs) < self.tube_len: #padding

            # last_idx = tube_frames_idxs[-1]
            # tube_frames_idxs += (self.tube_len - len(tube_frames_idxs))*[last_idx]
            # tube_frames_idxs = tube_frames_idxs
            # print('len(tube_frames_idxs) < self.tube_len: ', tube_frames_idxs)
            center_idx = int(len(tube_frames_idxs)/2)
            
            # print('center_idx:{}, center_frame:{}'.format(center_idx, tube_frames_idxs[center_idx]))
            
            start = tube_frames_idxs[center_idx]-int(self.tube_len/2)
            end = tube_frames_idxs[center_idx]+int(self.tube_len/2)
            # print('start: {}, end: {}'.format(start,end))
            out = list(range(start,end))
            # if tube_frames_idxs[center_idx]-int(self.tube_len/2) < self.max_video_len:
            if out[0]<0:
                most_neg = abs(out[0])
                out = [i+most_neg for i in out]
            elif tube_frames_idxs[center_idx]+int(self.tube_len/2) > max_video_len:
                start = tube_frames_idxs[center_idx]-(self.tube_len-(max_video_len-tube_frames_idxs[center_idx]))+1
                end = max_video_len+1
                out = list(range(start,end))
            tube_frames_idxs = out
            return tube_frames_idxs
    
    def __centered_segments__(self):
        """
        the overlap could cause less than 16 frames per tube
        """
        segment_size = 5
        stride = 1
        overlap = 0.5
        overlap_length = int(overlap*segment_size)
        total_len = self.tube_len*(segment_size-overlap_length)

        indices = [x for x in range(0, total_len, stride)]
        indices_segments = [indices[x:x + segment_size] for x in range(0, len(indices), segment_size-overlap_length)]
        indices_segments = [s for s in indices_segments if len(s)==segment_size]

        m = 75
        frames = list(range(150))
        frames = frames[m-int(total_len/2) : m+int(total_len/2)]
        video_segments = []
        for i, indices_segment in enumerate(indices_segments): #Generate segments using indices
            segment = np.asarray(frames)[indices_segment].tolist()
            video_segments.append(segment)
        return video_segments

    def __central_bbox__(self, tube, id):
        width, height = 224, 224
        if len(tube)>2:
            central_box = tube[int(len(tube)/2)]
        else:
            central_box = tube[0]
        central_box = central_box[0:4]
        central_box = np.array([max(central_box[0], 0), max(central_box[1], 0), min(central_box[2], width - 1), min(central_box[3], height - 1)])
        central_box = np.insert(central_box[0:4], 0, id).reshape(1,-1)
        central_box = torch.from_numpy(central_box).float()
        return central_box
