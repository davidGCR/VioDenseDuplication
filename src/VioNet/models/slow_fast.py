import torch
from typing import Dict
import json
import urllib
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)

from torchsummary import summary


if __name__=="__main__":
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
    device = "cpu"
    model = model.eval()
    model = model.to(device)

    # print(model)
    summary(model, [(3, 8, 256, 256), (3, 32, 256, 256)])
