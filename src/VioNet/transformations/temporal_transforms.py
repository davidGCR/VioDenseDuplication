import random
import numpy as np
import os
import pandas as pd
import re
from PIL import Image
from torchvision import transforms
import torch
from operator import itemgetter

def crop(frames, start, size, stride):
    # todo more efficient
    # padding by loop
    while start + (size - 1) * stride > len(frames) - 1:
        frames *= 2
    return frames[start:start + (size - 1) * stride + 1:stride]


class BeginCrop(object):
    def __init__(self, size, stride=1):
        self.size = size
        self.stride = stride

    def __call__(self, frames):
        return crop(frames, 0, self.size, self.stride)


class CenterCrop(object):
    def __init__(self, size, stride=1, input_type="rgb"):
        self.size = size
        self.stride = stride
        self.input_type = input_type

    def __call__(self, frames):
        start = max(0, len(frames) // 2 - self.size * self.stride // 2)
        crop_r = crop(frames, start, self.size, self.stride)
        if self.input_type == "rgb":
          return crop_r
        elif self.input_type == "dynamic-images":
          return [crop_r] 

class SequentialCrop(object):
    def __init__(self, size, stride=1, overlap=0.5, max_segments=0):
        self.size = size
        self.stride = stride
        self.overlap = overlap
        self.overlap_length = int(self.overlap*self.size)
        self.max_segments = max_segments

    def __call__(self, frames):
        indices = [x for x in range(0, len(frames), self.stride)]
        
        indices_segments = [indices[x:x + self.size] for x in range(0, len(indices), self.size-self.overlap_length)]
        # indices_segments = self.padding(indices_segments)
        # print('indices_segments:', len(indices_segments), indices_segments)

        video_segments = []
        for i, indices_segment in enumerate(indices_segments): #Generate segments using indices
            segment = np.asarray(frames)[indices_segment].tolist()
            video_segments.append(segment)
        video_segments = video_segments[0:self.max_segments] if self.max_segments > 0 else video_segments
        return video_segments

class RandomCrop(object):
    def __init__(self, size, stride=1, input_type="rgb"):
        self.size = size
        self.stride = stride
        self.input_type = input_type

    def __call__(self, frames):
        start = random.randint(
            0, max(0,
                   len(frames) - 1 - (self.size - 1) * self.stride)
        )
        crop_r = crop(frames, start, self.size, self.stride)
        if self.input_type == "rgb":
          return crop_r
        elif self.input_type == "dynamic-images":
          return [crop_r]


class OneFrameCrop(object):
    """
        Get only one frame of a video
    """
    def __init__(self, position=0): # 0: central, 1: random
        self.position = position

    def __call__(self, frames):
        frame = int(len(frames)/2) if self.position==0 else random.randint(1, len(frames))
        return [frame]

class TrainGuidedKeyFrameCrop(object):
    def __init__(self, size, segment_size=15, stride=1, overlap=0.5):
        self.size = size
        self.segment_size = segment_size
        self.stride = stride
        self.overlap = overlap
        self.overlap_length = int(self.overlap*self.size)
    
    def __call__(self, frames, tmp_annotation):
        if tmp_annotation == None:
            start = random.randint(
                0, max(0,
                    len(frames) - 1 - (self.size - 1) * self.stride)
                )
            # print(frames)
            return [crop(frames, start, self.size, self.stride)]
        else:
            df = pd.read_csv(tmp_annotation)
            df.sort_values(by = 'violence', inplace=True, ascending=False)
            _, v_name = os.path.split(df.iloc[0]["imgpath"])
            frame = int(re.search(r'\d+', v_name).group())
            left_limit = frame - int(self.segment_size / 2)
            right_limit = frame + int(self.segment_size / 2)
            start = left_limit if frame - int(self.segment_size / 2) > 0 else 1
            end = right_limit if frame + int(self.segment_size / 2) <= len(frames) else len(frames)

            # print('frame:{}, start:{}, end:{}'.format(frame, start, end))

            frames = []
            for i in range(start,end,1):
                # print(v_name, frame)
                frames.append(i)
            print(frames)
            return [frames]

class ValGuidedKeyFrameCrop(object):
    def __init__(self, size, segment_size=15, stride=1, overlap=0.5):
        self.size = size
        self.segment_size = segment_size
        self.stride = stride
        self.overlap = overlap
        self.overlap_length = int(self.overlap*self.size)
    
    def __call__(self, frames, tmp_annotation):
        df = pd.read_csv(tmp_annotation)
        df.sort_values(by = 'violence', inplace=True, ascending=False)
        _, v_name = os.path.split(df.iloc[0]["imgpath"])
        frame = int(re.search(r'\d+', v_name).group())
        left_limit = frame - int(self.segment_size / 2)
        right_limit = frame + int(self.segment_size / 2)
        start = left_limit if frame - int(self.segment_size / 2) > 0 else 1
        end = right_limit if frame + int(self.segment_size / 2) <= len(frames) else len(frames)

        # print('frame:{}, start:{}, end:{}'.format(frame, start, end))

        frames = []
        for i in range(start,end,1):
            # print(v_name, frame)
            frames.append(i)
        return [frames]
    

class KeyFrameCrop(object):
    """
    Key frame selection in positive and negative samples
    """
    def __init__(self, size, stride=1, input_type="rgb", dataset="train", group="larger"):
        self.size = size
        self.stride = stride
        self.input_type = input_type
        self.dataset = dataset
        self.group = group

    def __call__(self, frames, tmp_annotation, label):
        df = pd.read_csv(tmp_annotation)
        if self.group == "larger":
            df.sort_values(by = 'violence', inplace=True, ascending=False)
        elif self.group == "smaller":
            df.sort_values(by = 'violence', inplace=True, ascending=True)
        frames = []
        dicts = []
        for i in range(self.size):
            _, v_name = os.path.split(df.iloc[i]["imgpath"])
            frame = int(re.search(r'\d+', v_name).group())
            frames.append(frame)
            dicts.append({"frame":frame , "score": float(df.iloc[i]["violence"])})
        frames.sort()
        dicts = sorted(dicts, key = lambda i: i["frame"], reverse=False)
        if self.input_type == "rgb":
            return frames, dicts
        elif self.input_type == "dynamic-images":
            return [frames], dicts
        

class KeySegmentCrop(object):
    """
    Key segment selection in positive and negative samples
    """
    def __init__(self, size, stride=1, input_type="rgb", segment_type="fuse3segments"):
        self.size = size
        self.stride = stride
        self.input_type = input_type
        self.segment_type = segment_type
        
    
    def __get_segments__(self, annotation):
        df = pd.read_csv(annotation)
        df_v = df.loc[df['pred'] == 1]
        
        segments = []
        # for index, row in df_v.iterrows():
        #     print("===>",index, row["imgpath"], row["pred"])
        #     segments.append([int(n) for n in row["imgpath"].split('-')])
        if df_v.shape[0] >= 5:
            df_v = df_v.copy(deep=True)
            df_v.sort_values(by = 'violence', inplace=True, ascending=False)
            for i in range(5):
                str_frames = df_v.iloc[i]["imgpath"]
                segments.append([int(n) for n in str_frames.split('-')])
        else:
            for index, row in df_v.iterrows():
                # print("===>",index, row["imgpath"], row["pred"])
                segments.append([int(n) for n in row["imgpath"].split('-')])
        
        if len(segments) == 0:
            df.sort_values(by = 'violence', inplace=True, ascending=False)
            for i in range(5): ## Choose 5 more violent
                str_frames = df.iloc[i]["imgpath"]
                segments.append([int(n) for n in str_frames.split('-')])
        return segments
    
    def __get_best_segment__(self, annotation):
        segments = []
        df = pd.read_csv(annotation)
        df.sort_values(by='score', inplace=True, ascending=False)
        str_frames = df.iloc[0]["imgpath"]
        segments.append([int(n) for n in str_frames.split('-')])
        return segments
    
    def __expand_segment__(self, flat, frames):
        start = flat[0]
        end = flat[len(flat)-1]
        if start+self.size < len(frames):
            sample = list(range(start, start+self.size))
        else:
            sample = list(range(end-self.size+1, end+1))
        return sample

    
    def __get_random_crop__(self, frames):
        start = random.randint(
                0, max(0,
                    len(frames) - 1 - (self.size - 1) * self.stride)
            )
        sample = crop(frames, start, self.size, self.stride)
        return sample

    def __call__(self, frames, tmp_annotation):
        if self.segment_type == "fuse3segments":
            segments = self.__get_segments__(tmp_annotation)
        elif self.segment_type == "highestscore":
            segments = self.__get_best_segment__(tmp_annotation)
        flat = list(np.unique(np.concatenate(segments).flat)) if len(segments)>0 else []
        flat.sort()
        # print(flat)
        sample = None
        if len(flat) < self.size:
            sample = self.__expand_segment__(flat, frames)
        elif len(flat) > self.size:
            sample =  random.sample(flat, self.size)
            sample.sort()
        elif len(flat) == self.size:
            sample = flat
        
        if self.input_type == "rgb":
            return sample
        elif self.input_type == "dynamic-images":
            return [sample]

class IntervalCrop(object):
    """
    split videos in N segments and get the central frame of each segment
    """
    def __init__(self, intervals_num=16, interval_len=7, overlap=0):
        self.intervals_num = intervals_num
        self.interval_len = interval_len
        self.c_idx = int(self.interval_len/2)
        self.overlap = overlap
        self.overlap_length = int(self.overlap*self.interval_len)
    
    def __call__(self, frames):
        indices = [x for x in range(0, len(frames), 1)]
        indices_segments = [indices[x:x + self.interval_len] for x in range(0, len(indices), self.interval_len-self.overlap_length)]
        
        video_segments = []
        for i, indices_segment in enumerate(indices_segments): #Generate segments using indices
            segment = np.asarray(frames)[indices_segment].tolist()
            if len(segment) == self.interval_len:
                video_segments.append(segment[self.c_idx])
        if len(video_segments)>self.intervals_num:
            video_segments = video_segments[:self.intervals_num]

        return video_segments



class SegmentsCrop(object):
    def __init__(self, size, segment_size=15, stride=1, overlap=0.5, padding=True, position="start"):
        """
        Args:
            size (int): number of segments
            segment_size (int): length of each segment
            stride (int): frames to skip into a segment
            overlap (float): overlapping between each segment
            padding (bool): cut or add segments to get 'size' segments for each sample
            position (str): In case we want one segmet from the beginning, middle and end of the video 
        """
        self.size = size
        self.segment_size = segment_size
        self.stride = stride
        self.overlap = overlap
        self.overlap_length = int(self.overlap*self.segment_size)
        self.padding = padding
        self.position = position
    
    def __remove_short_segments__(self, segments):
        segments = [s for s in segments if len(s)==self.segment_size]
        return segments
    
    def __padding__(self, segment_list):
        if  len(segment_list) < self.size:
            idx = len(segment_list) - 1
            last_element = segment_list[idx]
            # while len(last_element) != self.segment_size and i >=0:
            #     i -= 1
            #     last_element = segment_list[i]

            for i in range(self.size - len(segment_list)):
                segment_list.append(last_element)
        # elif len(segment_list) > self.size:
        #     segment_list = segment_list[0:self.size]
        return segment_list
    
    def __choose_one_position__(self, segments):
        if self.position == "start":
            return segments[0]
        elif self.position == "middle":
            return segments[int(len(segments)/2)]
        elif self.position == "random":
            start = random.randint(0, len(segments)-1)
            while len(segments[(start)]) != self.segment_size:
                start = random.randint(0, len(segments)-1)
            
            # print(segments[start], len(segments))
            return segments[(start)]
    
    def __choose_multiple_positions__(self, segments):
        if self.position == "start":
            return segments[0:self.size]
        elif self.position == "middle":
            # segments = [s for s in segments if len(s)==self.size]
            c = int(len(segments)/2)
            start = c-int(self.size/2)
            end = c+int(self.size/2)
            idxs = [i for i in range(start,end+1)] if end-start < self.size else [i for i in range(start,end)]
            segments = list(itemgetter(*idxs)(segments))
            return segments
        elif self.position == "random":
            # segments = [s for s in segments if len(s)==self.size]
            segments = random.sample(segments, self.size)
            segments.sort()
            return segments

    def __call__(self, frames):
        
        indices = [x for x in range(0, len(frames), self.stride)]
        
        indices_segments = [indices[x:x + self.segment_size] for x in range(0, len(indices), self.segment_size-self.overlap_length)]

        indices_segments = self.__remove_short_segments__(indices_segments)
        
        if len(indices_segments)<self.size and self.padding:
            indices_segments = self.__padding__(indices_segments)
        if self.size == 1:
            indices_segments = [self.__choose_one_position__(indices_segments)]
        elif self.size > 1:
            indices_segments = self.__choose_multiple_positions__(indices_segments)
        elif self.size == 0: #all
            indices_segments = indices_segments
           
        video_segments = []
        for i, indices_segment in enumerate(indices_segments): #Generate segments using indices
            segment = np.asarray(frames)[indices_segment].tolist()
            video_segments.append(segment)
        # print(video_segments)
        return video_segments

class RandomSegmentsCrop(object):
    def __init__(self, size, segment_size=15, stride=1, overlap=0.5):
        self.size = size
        self.segment_size = segment_size
        self.stride = stride
        self.overlap = overlap
        self.overlap_length = int(self.overlap*self.size)
    
    def padding(self, segment_list):
        if  len(segment_list) < self.size:
            last_element = segment_list[len(segment_list) - 1]
            for i in range(self.size - len(segment_list)):
                segment_list.append(last_element)
        elif len(segment_list) > self.size:
            segment_list = segment_list[0:self.size]
        return segment_list

    def __call__(self, frames):
        indices = list(range(0, len(frames)))
        starts = random.sample(indices[:-self.segment_size], self.segment_size)
        starts.sort()
        segment0 = random.sample(indices[starts[0]:], 2*self.segment_size)
        segment0.sort()
        indices_segments = [segment0]
        for i in range(1, len(starts),1):
            segment = random.sample(indices[starts[i]:], self.segment_size)
            segment.sort()
            indices_segments.append(segment)

        video_segments = []
        for i, indices_segment in enumerate(indices_segments): #Generate segments using indices
            segment = np.asarray(frames)[indices_segment].tolist()
            video_segments.append(segment)
    
        return video_segments

import torch
# from transformations.transf_util import imread, PIL2tensor
# class Segment2Images(object):
#     def __init__(self, order):
#         self.order = order #(T, H, W, C)

#     def __call__(self, paths):
#         frames = [PIL2tensor(imread(p)) for p in paths]
#         video = torch.stack(frames, dim=0)
#         video = video.permute(0,2,3,1).type(torch.uint8)
#         return video




if __name__ == '__main__':
    # temp_transform = KeyFrameCrop(size=30, stride=1, input_type='rgb', group="larger")
    # frames = list(range(1, 150))
    # frames = temp_transform(frames)
    # print('Video video_segments:\n', len(video_segments), '\n',video_segments)
    # temp_transform = KeyFrameCrop(size=16, stride=1)
# 
    # temp_transform = SequentialCrop(size=5,stride=1,overlap=0.7, max_segments=16)
    # temp_transform = KeySegmentCrop(size=16,stride=1,input_type="rgb", segment_type="highestscore")
    temp_transform = SegmentsCrop(size=6, segment_size=10, stride=1,overlap=0,padding=True, position='middle')
    # temp_transform = IntervalCrop(intervals_num=16, interval_len=5, overlap=0.7)
    frames = list(range(1, 121))

    # frames = temp_transform(frames, "/Users/davidchoqueluqueroman/Documents/CODIGOS/AVSS2019/src/VioNet/v4dhdnsxiX4_1.csv", 0)
    # frames = temp_transform(frames, "/Users/davidchoqueluqueroman/Documents/CODIGOS/AVSS2019/src/VioNet/fbtEhNq5a6E_0.csv")
    # frames = temp_transform(frames, "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/rwf-vscores/val/NonFight/NeyOxUHJ_0.csv")
    # temp_transform = CenterCrop(size=16, stride=1)
    # temp_transform = RandomCrop(size=16, stride=1)
    frames = temp_transform(frames)
    print(frames)
    print(len(frames))
