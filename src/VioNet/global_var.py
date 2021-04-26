import sys
import os
dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

folders  = dirname.split(os.sep)
# print(folders)

def getFolder(specific_folder):
  if folders[1] == 'content':
      folder2save = os.path.join("/content/drive/My Drive/VIOLENCE DATA", specific_folder)
  elif folders[1] == 'Users':
      folder2save = os.path.join("/Users/davidchoqueluqueroman/Documents/CODIGOS/AVSS2019", specific_folder)
  return folder2save


FEAT_EXT_RESNEXT = "resnetxt"
FEAT_EXT_S3D = "s3d"
FEAT_EXT_C3D = "c3d"
FEAT_EXT_RESNEXT_S3D = "resnetxt+s3d"
FEAT_EXT_RESNET = "resnet"

RWF_DATASET = "rwf-2000"
HOCKEY_DATASET = "hockey"
VIF_DATASET = "vif"
# RWF_DATASET = "RWF-2000"

DYN_IMAGE = "dynamic-image"
RGB_FRAME = "rgb"
