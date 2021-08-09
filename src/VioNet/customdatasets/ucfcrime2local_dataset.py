import os
from re import split
import torch.utils.data as data
from TubeletGeneration.metrics import extract_tubes_from_video
from TubeletGeneration.tube_utils import JSON_2_videoDetections
from VioNet.customdatasets.make_dataset import MakeUCFCrime2LocalClips

class UCFCrime2LocalDataset(data.Dataset):
    """
    Load tubelets from one video
    Use to extract features tube-by-tube from just a video
    """

    def __init__(
        self, 
        root,
        path_annotations,
        abnormal,
        persons_detections_path,
        transform=None,
        clip_len=25,
        clip_temporal_stride=1):
        # self.dataset_root = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips',
        # self.split = 'anomaly',
        # self.video = 'Arrest036(2917-3426)',
        # self.p_d_path = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/ucfcrime2local',
        # self.gt_ann_path = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CrimeViolence2LocalDATASET/Txt annotations-longVideos'

        # self.dataset_root = dataset_root
        # self.split = split
        # self.video = video
        # self.p_d_path = p_d_path
        # self.gt_ann_path = gt_ann_path
        # self.transform = transform
        # self.person_detections = JSON_2_videoDetections(p_d_path)
        # self.tubes = extract_tubes_from_video(
        #     self.dataset_root,
        # )
        self.clip_temporal_stride = clip_temporal_stride
        self.clip_len = clip_len
        self.root = root
        self.path_annotations = path_annotations
        self.abnormal = abnormal
        self.make_dataset = MakeUCFCrime2LocalClips(root, path_annotations, abnormal)
        self.paths, self.labels, self.annotations = self.make_dataset()

        self.persons_detections_path = persons_detections_path

    def __len__(self):
        return len(self.paths)
    
    def get_video_clips(self, video_folder):
        _frames = os.listdir(video_folder)
        _frames = [f for f in _frames if '.jpg' in f]
        num_frames = len(_frames)
        indices = [x for x in range(0, num_frames, self.clip_temporal_stride)]
        indices_segments = [indices[x:x + self.clip_len] for x in range(0, len(indices), self.clip_len)]

        return indices_segments

    def generate_tube_proposals(self, path, frames):
        tmp = path.split('/')
        split = tmp[-2]
        video = tmp[-1]
        p_d_path = os.path.join(self.persons_detections_path, split, video)
        person_detections = JSON_2_videoDetections(p_d_path)
        tubes = extract_tubes_from_video(
            self.root,
            person_detections,
            frames,
            # {'wait': 200}
            )
        return tubes

    def __getitem__(self, index):
        path = self.paths[index]
        ann = self.annotations[index]
        spatial_annotations_gt = self.make_dataset.ground_truth_boxes(path, ann)

        return box, tube_images