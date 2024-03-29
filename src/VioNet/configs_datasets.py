# from caffe2.python.workspace import CreateBlob
from  config import Config
from  global_var import *
from utils import get_torch_device
from transformations.spatial_transforms import Normalize
from transformations.temporal_transforms import *
import torchvision.transforms as transforms

device = get_torch_device()

configs = {
        'hockey': {
            'lr': 1e-2,
            'batch_size': 8
        },
        'movie': {
            'lr': 1e-3,
            'batch_size': 16
        },
        'vif': {
            'lr': 1e-3,
            'batch_size': 16
        },
        'rwf-2000': {
            'lr': 1e-3,
            'batch_size': 8
        }
    }
environment_config = {
    'home': HOME_UBUNTU
}

def build_temporal_transformation(config: Config, transf_type: str, split='train'):
    if transf_type == STANDAR_CROP:
        temporal_transform = RandomCrop(size=config.sample_duration, stride=config.stride, input_type=config.input_type)
    elif transf_type == SEGMENTS_CROP:
        if split == 'train':
            temporal_transform = SegmentsCrop(size=config.sample_duration,
                                                segment_size=config.segment_size,
                                                stride=config.stride,
                                                overlap=config.overlap,
                                                position=config.train_sampling_type)
        else:
            temporal_transform = SegmentsCrop(size=config.sample_duration,
                                            segment_size=config.segment_size,
                                            stride=config.stride,
                                            overlap=config.overlap,
                                            position=config.val_sampling_type)

    elif transf_type == CENTER_CROP:
        temporal_transform = CenterCrop(size=config.sample_duration, stride=config.stride, input_type=config.input_type)
    elif transf_type == KEYFRAME_CROP:
        temporal_transform = KeyFrameCrop(size=config.sample_duration, stride=config.stride, input_type=config.input_type)
    elif transf_type == GUIDED_KEYFRAME_CROP:
        temporal_transform = TrainGuidedKeyFrameCrop(size=config.sample_duration, segment_size=config.segment_size, stride=config.stride, overlap=0.5)
    elif transf_type == KEYSEGMENT_CROP:
        temporal_transform = KeySegmentCrop(size=config.sample_duration, stride=config.stride, input_type=config.input_type, segment_type="highestscore")
    elif transf_type == INTERVAL_CROP:
        temporal_transform = IntervalCrop(intervals_num=config.sample_duration, interval_len=config.segment_size)
    return temporal_transform



def hockey_i3d_config():
    """
    Sample 16 rgb or DI from 40 frames video
    """
    config = Config(
        'i3d',  # c3d, convlstm, densenet, densenet_lean, resnet50, densenet2D
        HOCKEY_DATASET,
        device=device,
        num_epoch=20,
        acc_baseline=0.95,
        ft_begin_idx=0
    )
    config.input_type = 'rgb' #rgb, dynamic-images
    config.train_temporal_transform = INTERVAL_CROP#SEGMENTS_CROP#INTERVAL_CROP
    config.val_temporal_transform = INTERVAL_CROP#SEGMENTS_CROP#INTERVAL_CROP
    config.sample_duration = 16 #number of segments
    config.segment_size = 5 #len of segments
    config.stride = 1
    config.overlap = 0.7

    config.train_batch = 8
    config.val_batch = 8
    config.learning_rate = 1e-2

    config.additional_info = ''
    return config

def hockey_MDIResNet_config():
    """
    Sample 16 rgb or DI from 40 frames video
    """
    config = Config(
        'MDIResNet',  # c3d, convlstm, densenet, densenet_lean, resnet50, densenet2D
        HOCKEY_DATASET,
        device=device,
        num_epoch=50,
        acc_baseline=0.95,
        ft_begin_idx=0
    )
    # config.sample_size = (224,224)

    config.input_type = 'dynamic-images' #rgb, dynamic-images
    config.train_temporal_transform = SEGMENTS_CROP#SEGMENTS_CROP#INTERVAL_CROP
    config.val_temporal_transform = SEGMENTS_CROP#SEGMENTS_CROP#INTERVAL_CROP
    config.sample_duration = 4 #number of segments
    config.segment_size = 10 #len of segments
    config.stride = 1
    config.overlap = 0

    config.train_batch = 8
    config.val_batch = 8
    config.learning_rate = 1e-2

    config.additional_info = ''
    return config

def rwf_config():
    """
    Sample 16 rgb or DI from 120 frames video
    """
    config = Config(
        'i3d',  # c3d, convlstm, densenet, densenet_lean, resnet50, densenet2D
        RWF_DATASET,
        device=device,
        num_epoch=15,
        acc_baseline=0.81,
        ft_begin_idx=0,
    )

    config.input_type = 'rgb' #rgb, dynamic-images
    config.train_temporal_transform = INTERVAL_CROP
    config.val_temporal_transform = INTERVAL_CROP
    config.sample_duration = 16 #number of segments
    config.segment_size = 7 #len of segments
    config.stride = 1
    config.overlap = 0
    # config.temp_annotation_path = os.path.join(environment_config['home'], PATH_SCORES,
    #     "Scores-dataset(rwf-2000)-ANmodel(AnomalyDetector_Dataset(UCFCrime2LocalClips)_Features(c3d)_TotalEpochs(100000)_ExtraInfo(c3d)-Epoch-10000)-input(rgb)")

    config.train_batch = 8
    config.val_batch = 8
    config.learning_rate = 1e-3

    config.additional_info = ''
    return config

def rwf_MDIResNet_config():
    """
    Sample 16 rgb or DI from 120 frames video
    """
    config = Config(
        'MDIResNet',  # c3d, convlstm, densenet, densenet_lean, resnet50, densenet2D
        RWF_DATASET,
        device=device,
        num_epoch=50,
        acc_baseline=0.87,
        ft_begin_idx=0,
    )

    config.input_type = 'dynamic-images' #rgb, dynamic-images
    config.train_temporal_transform = SEGMENTS_CROP
    config.val_temporal_transform = SEGMENTS_CROP
    config.sample_duration = 6 #number of segments
    config.segment_size = 10 #len of segments
    config.stride = 1
    config.overlap = 0
    config.train_sampling_type = 'random'
    config.val_sampling_type = 'middle'

    config.train_batch = 8
    config.val_batch = 8
    config.learning_rate = 1e-3

    config.additional_info = ''
    return config


def rwf_twostream_config():
    """
    Sample 16 rgb
    """
    config = Config(
        'TwoStreamVD_Binary_CFam',  # c3d, convlstm, densenet, densenet_lean, resnet50, densenet2D
        RWF_DATASET,
        device=device,
        num_epoch=50,
        acc_baseline=0.87,
    )

    config.input_type = 'rgb' #rgb, dynamic-images
    config.train_temporal_transform = STANDAR_CROP
    config.val_temporal_transform = CENTER_CROP
    config.sample_duration = 1 #number of segments
    config.segment_size = 16 #len of segments
    config.stride = 1
    config.overlap = 0

    
    # config.train_sampling_type = 'random'
    # config.val_sampling_type = 'middle'

    config.train_batch = 8
    config.val_batch = 8
    config.learning_rate = 1e-3

    config.additional_info = ''
    return config