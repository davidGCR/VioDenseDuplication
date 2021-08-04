import torch
import torch.nn as nn

class Identity(nn.Module):
  def __init__(self):
      super().__init__()

  def forward(self, x):
      return x

class Backbone3DResNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    self.backbone.blocks[4] = Identity()
    self.backbone.blocks[5] = Identity()
    # self.roi_layer = RoiPoolLayer()
  
  def forward(self, x):
    x = self.backbone(x)
    # x = self.roi_layer(x, bbox)
    return x