import random
import numpy as np
import os
import pandas as pd
import re

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
    def __init__(self, size, stride=1):
        self.size = size
        self.stride = stride

    def __call__(self, frames):
        start = max(0, len(frames) // 2 - self.size * self.stride // 2)
        
        return crop(frames, start, self.size, self.stride)


class RandomCrop(object):
    def __init__(self, size, stride=1):
        self.size = size
        self.stride = stride

    def __call__(self, frames):
        start = random.randint(
            0, max(0,
                   len(frames) - 1 - (self.size - 1) * self.stride)
        )
        # print('RandomCrop startr:', start)
        return crop(frames, start, self.size, self.stride)

class KeyFrameCrop(object):
    def __init__(self, size, stride=1):
        self.size = size
        self.stride = stride

    def __call__(self, frames, tmp_annotation):
        if tmp_annotation == None:
            start = random.randint(
                0, max(0,
                    len(frames) - 1 - (self.size - 1) * self.stride)
                )
            return crop(frames, start, self.size, self.stride)
        else:
            df = pd.read_csv(tmp_annotation)
            df.sort_values(by = 'violence', inplace=True, ascending=False)
            frames = []
            for i in range(self.size):
                _, v_name = os.path.split(df.iloc[i]["imgpath"])
                frame = int(re.search(r'\d+', v_name).group())
                # print(v_name, frame)
                frames.append(frame)
            frames.sort()
            return frames

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
            return crop(frames, start, self.size, self.stride)
        else:
            df = pd.read_csv(tmp_annotation)
            df.sort_values(by = 'violence', inplace=True, ascending=False)
            _, v_name = os.path.split(df.iloc[0]["imgpath"])
            frame = int(re.search(r'\d+', v_name).group())
            left_limit = frame - int(self.segment_size / 2)
            right_limit = frame + int(self.segment_size / 2)
            start = left_limit if frame - int(self.segment_size / 2) > 0 else 1
            end = right_limit if frame + int(self.segment_size / 2) <= len(frames) else len(frames)

            print('frame:{}, start:{}, end:{}'.format(frame, start, end))

            frames = []
            for i in range(start,end,1):
                # print(v_name, frame)
                frames.append(i)
            return frames

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

        print('frame:{}, start:{}, end:{}'.format(frame, start, end))

        frames = []
        for i in range(start,end,1):
            # print(v_name, frame)
            frames.append(i)
        return frames
    

class TrainKeyFrameCrop(object):
    """ 
    Key frame selection in positive samples and random in negative samples
    Use for Dynamic images
    """
    def __init__(self, size, stride=1):
        self.tt = KeyFrameCrop(size, stride)

    def __call__(self, frames, tmp_annotation):
        frames = self.tt(frames, tmp_annotation)
        return [frames]

class ValKeyFrameCrop(object):
    """
    Key frame selection in positive and negative samples
    Use for Dynamic images
    """
    def __init__(self, size, stride=1, input_type="rgb"):
        self.size = size
        self.stride = stride
        self.input_type = input_type

    def __call__(self, frames, tmp_annotation):
        df = pd.read_csv(tmp_annotation)
        df.sort_values(by = 'violence', inplace=True, ascending=False)
        frames = []
        for i in range(self.size):
            _, v_name = os.path.split(df.iloc[i]["imgpath"])
            frame = int(re.search(r'\d+', v_name).group())
            # print(v_name, frame)
            frames.append(frame)
        frames.sort()
        if self.input_type == "rgb":
            return frames
        elif self.input_type == "dynamic-images":
            return [frames]


class SegmentsCrop(object):
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
        
        indices = [x for x in range(0, len(frames), self.stride)]
        
        indices_segments = [indices[x:x + self.segment_size] for x in range(0, len(indices), self.segment_size-self.overlap_length)]
        indices_segments = self.padding(indices_segments)
        # print('indices_segments:', len(indices_segments), indices_segments)

        video_segments = []
        for i, indices_segment in enumerate(indices_segments): #Generate segments using indices
            segment = np.asarray(frames)[indices_segment].tolist()
            video_segments.append(segment)
        
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
        


if __name__ == '__main__':
    temp_transform = GuidedKeyFrameCrop(size=1, segment_size=30, stride=1, overlap=0)
    # frames = list(range(1, 150))
    # frames = temp_transform(frames)
    # print('Video video_segments:\n', len(video_segments), '\n',video_segments)
    # temp_transform = KeyFrameCrop(size=16, stride=1)
    
    frames = list(range(1, 150))

    frames = temp_transform(frames, '/Users/davidchoqueluqueroman/Documents/CODIGOS/protest-detection-violence-estimation/rwf_predictions/train/Fight/oIEZ45OCmAw_3.csv')
    # temp_transform = CenterCrop(size=16, stride=1)
    # temp_transform = RandomCrop(size=16, stride=1)
    # frames = temp_transform(frames)
    print(frames)
