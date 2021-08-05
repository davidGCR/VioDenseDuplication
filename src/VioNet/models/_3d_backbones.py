import torch
import torch.nn as nn
from models.i3d import InceptionI3d

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

class BackboneI3D(nn.Module):
  def __init__(self, final_endpoint, pretrained, freeze=False):
    super().__init__()
    self.backbone = InceptionI3d(2, in_channels=3, final_endpoint=final_endpoint)
    load_model_path = pretrained
    state_dict = torch.load(load_model_path)
    self.backbone.load_state_dict(state_dict,  strict=False)
    if freeze:
      for param in self.backbone.parameters():
          param.requires_grad = False
  
  def forward(self, x):
    x = self.backbone(x)
    return x
    


