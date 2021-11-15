import sys
import os
dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

folders  = dirname.split(os.sep)
# print(folders)

HOME_UBUNTU = "/media/david/datos/Violence DATA/"
HOME_DRIVE = "/content/drive/My Drive/VIOLENCE DATA"
HOME_COLAB = "/content/DATASETS"
HOME_OSX = "/Users/davidchoqueluqueroman/Documents/DATASETS_Local"

FEAT_EXT_RESNEXT = "resnetxt"
FEAT_EXT_S3D = "s3d"
FEAT_EXT_C3D = "c3d"
FEAT_EXT_RESNEXT_S3D = "resnetxt+s3d"
FEAT_EXT_RESNET = "resnet"

RWF_DATASET = "rwf-2000"
RLSV_DATASET = "RealLifeViolenceDataset"
HOCKEY_DATASET = "hockey"
RLVSD_DATASET = "RealLifeViolenceDataset"
UCFCrime_DATASET = "UCFCrime"
UCFCrimeReduced_DATASET = "UCFCrime_Reduced"
VIF_DATASET = "vif"
VIO_DB_DATASETS = "VioNetDB-splits"
VIONET_WEIGHTS = "VioNet_weights"
UCFCrime2Local_DATASET = "UCFCrime2Local"
UCFCrime2LocalClips_DATASET = "UCFCrime2LocalClips"




PATH_TENSORBOARD = "VioNet_tensorboard_log"
PATH_LOG= "VioNet_log"
PATH_CHECKPOINT = "VioNet_pth"
PATH_SCORES = "KeysegmentScores"
MODEL_ANOMALY_DET = "AnomalyDetector"
# def getFolder(specific_folder):
#   if folders[1] == 'content':
#       folder2save = os.path.join("/content/drive/My Drive/VIOLENCE DATA", specific_folder)
#   elif folders[1] == 'Users':
#       folder2save = os.path.join("/Users/davidchoqueluqueroman/Documents/CODIGOS/AVSS2019", specific_folder)
#   return folder2save


# TEMPORAL_TRANSFORMATION NAMES
STANDAR_CROP = 'standar'
SEGMENTS_CROP = 'segments-crop' #for dynamic images
CENTER_CROP = 'center-crop'
KEYFRAME_CROP = 'keyframe'
GUIDED_KEYFRAME_CROP = 'guided-segment'
KEYSEGMENT_CROP = 'keysegment'
INTERVAL_CROP = 'interval-crop'

#MODEL NAMES
I3D = 'i3d'
MDIResNet = 'MDIResNet'

REGRESSION = 'regression'
BINARY = 'binary'

#tube sample strategies
MIDDLE_FRAMES = 'middle'
EVENLY_FRAMES = 'evenly'
# CENTER_FRAMES = 'center'

#box tube
MIDDLE_BOX = 0
UNION_BOX = 1
ALL_BOX = 2

#keyframe pos
MIDDLE_KEYFRAME = 0
BEGIN_KEYFRAME = 1
RANDOM_KEYFRAME = 2

DYN_IMAGE = "dynamic-image"
RGB_FRAME = "rgb"