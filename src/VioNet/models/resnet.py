import torch.nn as nn
from torchvision import models
import torch

class Identity(nn.Module):
  def __init__(self):
      super().__init__()

  def forward(self, x):
      return x

class ResNet(nn.Module):
    def __init__(self, num_classes = 2,
                        numDiPerVideos = 1, 
                        model_name = 'resnet50'):
        super(ResNet, self).__init__()
        self.numDiPerVideos = numDiPerVideos
        self.num_classes = num_classes
        # self.joinType = joinType
        model_ft = None
        if model_name == 'resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif model_name == 'resnet34':
            model_ft = models.resnet34(pretrained=True)
        elif model_name == 'resnet50':
            model_ft = models.resnet50(pretrained=True)
        
        self.num_ftrs = model_ft.fc.in_features
        model_ft.fc = Identity()
        if model_name == 'resnet18' or model_name == 'resnet34':
            self.bn = nn.BatchNorm2d(512)
        elif model_name == 'resnet50':
            self.bn = nn.BatchNorm2d(2048)
        
        self.convLayers = nn.Sequential(*list(model_ft.children())[:-2])  # to tempooling
        model_ft = None
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1,1))

        # set_parameter_requires_grad(self.convLayers, freezeConvLayers)
        
        if model_name == 'resnet18' or model_name == 'resnet34':
            self.linear = nn.Linear(512, self.num_classes)
        elif model_name == 'resnet50':
            self.linear = nn.Linear(2048, self.num_classes)
           

    def forward(self, X):
        # (ipts, vid_name, dynamicImages, one_box)
        # (x, vid_name, dynamicImages, bboxes) = X
        # print('X input:', X.size())
        # batch_size, C, timesteps, H, W = x.size()
        # c_in = x.view(batch_size * timesteps, C, H, W)
        x = torch.squeeze(X)
        x = self.convLayers(x)  #torch.Size([8, 2048, 7, 7]
        
        x = self.bn(x) # torch.Size([8, 2048, 7, 7])

        x = self.AdaptiveAvgPool2d(x) #torch.Size([8, 2048, 1, 1])
        # print('AdaptiveAvgPool2d: ', x.size())
        x = torch.flatten(x, 1)
        # num_fc_input_features = self.linear.in_features
        # x = x.view(batch_size, timesteps, num_fc_input_features)
        # x = x.max(dim=1).values
        x = self.linear(x)
        return x