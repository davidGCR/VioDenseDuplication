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
                    # boxes_sample_strategy=MIDDLE,
                    random=True,
                    box_as_tensor=True,
                    shape=(224,224)):
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
        # self.boxes_sample_strategy = boxes_sample_strategy
        self.box_as_tensor = box_as_tensor
        self.shape = shape

    def __call__(self, tubes: list,):
        segments = []
        # boxes = []
        if self.random: # select randomly K tubes
            chosed_tubes_idxs = random.sample(range(len(tubes)), self.max_num_tubes)
            # print('\nchosed_tubes_idxs:', chosed_tubes_idxs)
            # chosed_tubes = list(itemgetter(*chosed_tubes_idxs)(tubes))
            chosed_tubes = [tubes[i] for i in range(len(tubes)) if i in chosed_tubes_idxs]
        else:
            chosed_tubes = tubes[0:self.max_num_tubes]
        
        # print('chosed_tubes: \n', chosed_tubes)

        for tube in chosed_tubes:
            if self.input_type==RGB_FRAME:
                frames_idxs = self.__sampled_tube_frames_indices__(tube['foundAt'])
            else:
                frames_idxs = self.__centered_segments__()
            
            # bboxes = self.__sampled_tube_bboxes__(tube['boxes'], frames_idxs)
            # boxes.append(bboxes)
            segments.append(frames_idxs)

        # for tube in tubes:
        #     if self.input_type==RGB_FRAME:
        #         frames_idxs = self.__centered_frames__(tube['foundAt'],max_video_len)
        #     else:
        #         frames_idxs = self.__centered_segments__()
        #     if len(frames_idxs) > 0:
        #         bbox = self.__central_bbox__(tube['boxes'], tube['id']+1)
        #         boxes.append(bbox)
        #         segments.append(frames_idxs)
        #         # print(
        #         #     '\ntube[foundAt]: ', tube['foundAt'] , 
        #         #         '\tsample: ', frames_idxs,
        #         #         '\tbox: ',bbox)   
        # idxs = range(len(boxes))
        # if self.max_num_tubes != 0 and len(boxes) > self.max_num_tubes:
        #     if self.random:
        #         idxs = random.sample(range(len(boxes)), self.max_num_tubes)
        #         boxes = list(itemgetter(*idxs)(boxes))
        #         # print('random boxes: ', boxes)
        #         boxes = [b.reshape(1,-1) for b in boxes]
        #         segments = list(itemgetter(*idxs)(segments))
        #         segments = [segments] if self.max_num_tubes==1 else segments
        #         # if self.train:
        #         #     idxs = random.sample(range(len(boxes)), self.max_num_tubes)
        #         #     boxes = list(itemgetter(*idxs)(boxes))
        #         #     segments = list(itemgetter(*idxs)(segments))
        #         # else:
        #         #     n = len(boxes)
        #         #     m = int(n/2)
        #         #     boxes = boxes[m-int(self.max_num_tubes/2) : m+int(self.max_num_tubes/2)]
        #         #     segments = segments[m-int(self.max_num_tubes/2) : m+int(self.max_num_tubes/2)]
        #     else:
        #         boxes = boxes[0:self.max_num_tubes]
        #         segments = segments[0:self.max_num_tubes]
        
        # if len(boxes)==1:
        #     boxes[0] = torch.unsqueeze(boxes[0], 0)
        # print('tube_crop boxes: ', boxes, boxes[0].shape)
        # for id,box in enumerate(boxes):
        #     boxes[id][0,0] = id
        # print('boxes ids: ', boxes)
        return segments, chosed_tubes
    
    def __sampled_tube_frames_indices__(self, tube_found_at: list):
        max_video_len = tube_found_at[-1]
        if len(tube_found_at) == self.tube_len: 
            return tube_found_at
        if len(tube_found_at) > self.tube_len:
            if self.sample_strategy == MIDDLE:
                n = len(tube_found_at)
                m = int(n/2)
                arr = np.array(tube_found_at)
                centered_array = arr[m-int(self.tube_len/2) : m+int(self.tube_len/2)]
            elif self.sample_strategy == EVENLY:
                min_frame = tube_found_at[0]
                tube_frames_idxs = np.linspace(min_frame, max_video_len, self.tube_len).astype(int)
                tube_frames_idxs = tube_frames_idxs.tolist()
            return centered_array.tolist()
        if len(tube_found_at) < self.tube_len: #padding
            min_frame = tube_found_at[0]
            tube_frames_idxs = np.linspace(min_frame, max_video_len, self.tube_len).astype(int)
            tube_frames_idxs = tube_frames_idxs.tolist()
            # center_idx = int(len(tube_frames_idxs)/2)
            # # print('center_idx:{}, center_frame:{}'.format(center_idx, tube_frames_idxs[center_idx]))
            # start = tube_frames_idxs[center_idx]-int(self.tube_len/2)
            # end = tube_frames_idxs[center_idx]+int(self.tube_len/2)
            # out = list(range(start,end))
            # if out[0]<0:
            #     most_neg = abs(out[0])
            #     out = [i+most_neg for i in out]
            # elif tube_frames_idxs[center_idx]+int(self.tube_len/2) > max_video_len:
            #     start = tube_frames_idxs[center_idx]-(self.tube_len-(max_video_len-tube_frames_idxs[center_idx]))+1
            #     end = max_video_len+1
            #     out = list(range(start,end))
            # tube_frames_idxs = out
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

    def __format_bbox__(self, bbox):
        """
        Format a tube bbox: [x1,y1,x2,y2] to a correct format
        """
        (width, height) = self.shape
        bbox = bbox[0:4]
        bbox = np.array([max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], width - 1), min(bbox[3], height - 1)])
        bbox = np.insert(bbox[0:4], 0, id).reshape(1,-1).astype(float)
        if self.box_as_tensor:
            bbox = torch.from_numpy(bbox).float()
        return bbox

    # def __sampled_tube_bboxes__(self, tube_boxes, frames_idxs):
    #     """
    #     Sample a set of bboxes from a tube
    #     """
    #     if self.__sampled_tube_bboxes__ == MIDDLE:
    #         if len(tube_boxes)>2:
    #             central_box = tube_boxes[int(len(tube_boxes)/2)]
    #         else:
    #             central_box = tube_boxes[0]
    #         return central_box
    #     else:
    #         sampled_boxes = [tube_boxes[i] for i in range(tube_boxes) if i in frames_idxs]
    #         return sampled_boxes
        
        
