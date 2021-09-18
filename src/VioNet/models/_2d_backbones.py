import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Dict, Optional, cast
from torch import Tensor
from collections import OrderedDict 
from torchvision.models.resnet import *
from torchvision.models.resnet import BasicBlock, Bottleneck

class IntResNet(ResNet):
    def __init__(self,output_layer,*args):
        self.output_layer = output_layer
        super().__init__(*args)
        
        self._layers = []
        for l in list(self._modules.keys()):
            # print(l)
            self._layers.append(l)
            if l == output_layer:
                break
        self.layers = OrderedDict(zip(self._layers,[getattr(self,l) for l in self._layers]))

    def _forward_impl(self, x):
        for l in self._layers:
            x = self.layers[l](x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

class Backbone2DResNet(nn.Module):
    # base_model : The model we want to get the output from
    # base_out_layer : The layer we want to get output from
    # num_trainable_layer : Number of layers we want to finetune (counted from the top)
    #                       if enetered value is -1, then all the layers are fine-tuned
    def __init__(self,base_model,base_out_layer,num_trainable_layers):
        super().__init__()
        self.base_model = base_model
        self.base_out_layer = base_out_layer
        self.num_trainable_layers = num_trainable_layers
        
        self.model_dict = {'resnet18':{'block':BasicBlock,'layers':[2,2,2,2],'kwargs':{}},
                           'resnet34':{'block':BasicBlock,'layers':[3,4,6,3],'kwargs':{}},
                           'resnet50':{'block':Bottleneck,'layers':[3,4,6,3],'kwargs':{}},
                           'resnet101':{'block':Bottleneck,'layers':[3,4,23,3],'kwargs':{}},
                           'resnet152':{'block':Bottleneck,'layers':[3,8,36,3],'kwargs':{}},
                           'resnext50_32x4d':{'block':Bottleneck,'layers':[3,4,6,3],
                                              'kwargs':{'groups' : 32,'width_per_group' : 4}},
                           'resnext101_32x8d':{'block':Bottleneck,'layers':[3,4,23,3],
                                               'kwargs':{'groups' : 32,'width_per_group' : 8}},
                           'wide_resnet50_2':{'block':Bottleneck,'layers':[3,4,6,3],
                                              'kwargs':{'width_per_group' : 64 * 2}},
                           'wide_resnet101_2':{'block':Bottleneck,'layers':[3,4,23,3],
                                               'kwargs':{'width_per_group' : 64 * 2}}}
        
        #PRETRAINED MODEL
        self.resnet = self.new_resnet(self.base_model,self.base_out_layer,
                                     self.model_dict[self.base_model]['block'],
                                     self.model_dict[self.base_model]['layers'],
                                     True,True,
                                     **self.model_dict[self.base_model]['kwargs'])

        self.layers = list(self.resnet._modules.keys())
        #FREEZING LAYERS
        self.total_children = 0
        self.children_counter = 0
        for c in self.resnet.children():
            self.total_children += 1
            
        if self.num_trainable_layers == -1:
            self.num_trainable_layer = self.total_children
        
        for c in self.resnet.children():
            if self.children_counter < self.total_children - self.num_trainable_layers:
                for param in c.parameters():
                    param.requires_grad = False
            else:
                for param in c.parameters():
                    param.requires_grad =True
            self.children_counter += 1
                    
    def new_resnet(self,
                   arch: str,
                   outlayer: str,
                   block: Type[Union[BasicBlock, Bottleneck]],
                   layers: List[int],
                   pretrained: bool,
                   progress: bool,
                   **kwargs: Any
                  ) -> IntResNet:

        model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
            'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
            'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
            'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
            'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
        }

        model = IntResNet(outlayer, block, layers, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  progress=progress)
            model.load_state_dict(state_dict)
        return model
    
    def forward(self,x):
        x = self.resnet(x)
        return x

from torchsummary import summary
from torchvision import models

if __name__ ==  '__main__':
    model = Backbone2DResNet('resnet50','fc',num_trainable_layers=3)
    print(model)
    # summary(model,input_size=(3, 224, 224))
    
    # model_ft = models.resnet50(pretrained=True)
    # summary(model_ft,input_size=(3, 224, 224))
    # print(model_ft)

    input = torch.rand(4,3,224,224)
    output=model(input)
    print('out: ', output.size())
    