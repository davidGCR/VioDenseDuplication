import torch.nn as nn
from torchvision import models
import torch

def get_fine_tuning_params(model, index_layer):
    if not index_layer == 0:
        return model.parameters()

    if index_layer == -1:
        for name, param in model.named_parameters():
            if name == 'classifier':
                param.requires_grad = True
                break
            else:
                param.requires_grad = False

        # for param in model.parameters():

        #     param.requires_grad = False
        return model.parameters()

    
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
            self.classifier = nn.Linear(512, self.num_classes)
        elif model_name == 'resnet50':
            self.classifier = nn.Linear(2048, self.num_classes)
           

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
        x = self.classifier(x)
        return x

class Densenet2D(nn.Module):  
    def __init__(self, num_classes = 2,
                        numDiPerVideos = 1):
        super(Densenet2D, self).__init__()
        self.numDiPerVideos = numDiPerVideos
        self.num_classes = num_classes
        self.model = models.densenet121(pretrained=True)
        self.num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.num_ftrs, num_classes)
        # self.linear= nn.Linear(self.num_ftrs, num_classes)
        
        # self.tmpPooling = nn.MaxPool2d((numDiPerVideos, 1))

    def forward(self, x):
        # print('densenet input:', x.size())
        # batch_size, timesteps, C, H, W = x.size()
        # c_in = x.view(batch_size * timesteps, C, H, W)
        x = torch.squeeze(x)
        # print('cin: ', c_in.size())
        x = self.model(x)
        # print('cout: ', c_out.size())
        # x = torch.flatten(c_out, 1)
        # print('flatten: ', x.size())
        # Re-structure the data and then temporal max-pool.
        # x = x.view(batch_size, timesteps, self.num_ftrs)
        # print('Re-structure: ', x.size())
        # x = x.max(dim=1).values
        # print('maxpooling: ', x.size())
        # x = self.linear(x)
        return x

if __name__ == '__main__':
    model = Densenet2D()
    params = get_fine_tuning_params(model, -1)
    # print(params)
    print(model)