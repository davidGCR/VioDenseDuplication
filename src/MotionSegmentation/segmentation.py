import os
import sys
g_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('main g_path:', g_path)
sys.path.insert(1, g_path)

from VioNet.transformations.dynamic_image_transformation import DynamicImage
from VioNet.transformations.temporal_transforms import SegmentsCrop

def segment(video_folder_path):
    DN = DynamicImage(output_type="pil")
    tmp_transform = SegmentsCrop(size=5, segment_size=10, stride=1, overlap=0, padding=True)
    frames_idx = range(len(120))
    return tmp_transform(frames_idx)

if __name__=='__main__':
    segments = segment('')