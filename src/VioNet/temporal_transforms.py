import random
import numpy as np


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
        # print('indices:', indices)
        # print('starts:', starts)

        # print('0000:' ,indices[starts[0]:])
        segment0 = random.sample(indices[starts[0]:], 2*self.segment_size)
        segment0.sort()
        indices_segments = [segment0]
        print(0, '\t',segment0)
        for i in range(1, len(starts),1):
            segment = random.sample(indices[starts[i]:], self.segment_size)
            segment.sort()
            # print(i,'start:',starts[i], '\t',segment)
            indices_segments.append(segment)

            

        
        
        # indices = [x for x in range(0, len(frames), self.stride)]
        
        # indices_segments = [indices[x:x + self.segment_size] for x in range(0, len(indices), self.segment_size-self.overlap_length)]
        # indices_segments = self.padding(indices_segments)
        # # print('indices_segments:', len(indices_segments), indices_segments)

        video_segments = []
        for i, indices_segment in enumerate(indices_segments): #Generate segments using indices
            segment = np.asarray(frames)[indices_segment].tolist()
            video_segments.append(segment)
    
        return video_segments
        


if __name__ == '__main__':
    temp_transform = RandomSegmentsCrop(size=16, segment_size=15, stride=1, overlap=0.5)
    frames = list(range(1, 150))
    video_segments = temp_transform(frames)
    print('Video video_segments:\n', len(video_segments), '\n',video_segments)
