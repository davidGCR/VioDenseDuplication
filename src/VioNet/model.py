import torch
import torch.nn as nn

import models.densenet as dn
from models.c3d import C3D
from models.densenet import densenet88, densenet121
from models.convlstm import ConvLSTM
from models.models2D import ResNet, Densenet2D
import os
import models.models2D as rn

g_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(g_path)

def VioNet_Densenet2D(config):
    device = config.device
    model = Densenet2D(num_classes=2).to(device)
    params = rn.get_fine_tuning_params(model, config.ft_begin_idx)
    return model, params

def VioNet_Resnet(config):
    device = config.device
    model = ResNet(num_classes=2).to(device)
    if config.pretrained_model:
        state_dict = torch.load(g_path +'/VioNet/weights/'+ config.pretrained_model)
        model.load_state_dict(state_dict)

    params = rn.get_fine_tuning_params(model, config.ft_begin_idx)
    return model, params

def VioNet_C3D(config):
    device = config.device
    model = C3D(num_classes=2).to(device)

    # print(model)

    state_dict = torch.load(g_path +'/VioNet/'+ 'weights/C3D_Kinetics.pth')
    
    # state_dict =state_dict['state_dict']

    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     if k[0:2] == 'fc':
    #         continue
    #     name = 'features.'+k # remove 'module.' of dataparallel
    #     print(k, '\t',name)
    #     new_state_dict[name]=v

    # model.load_state_dict(new_state_dict, strict=False)
    
    model.load_state_dict(state_dict, strict=False)
    params = model.parameters()

    return model, params


def VioNet_ConvLSTM(config):
    device = config.device
    model = ConvLSTM(256, device).to(device)
    # freeze pretrained alexnet params
    for name, param in model.named_parameters():
        if 'conv_net' in name:
            param.requires_grad = False
    params = model.parameters()

    return model, params


def VioNet_densenet(config):
    device = config.device
    ft_begin_idx = config.ft_begin_idx
    sample_size = config.sample_size[0]
    sample_duration = config.sample_duration

    model = densenet121(num_classes=2,
                        sample_size=sample_size,
                        sample_duration=sample_duration).to(device)

    state_dict = torch.load(g_path +'/VioNet/'+ 'weights/DenseNet_Kinetics.pth')
    model.load_state_dict(state_dict)

    params = dn.get_fine_tuning_params(model, ft_begin_idx)

    return model, params


# the model we finally adopted in DenseNet
def VioNet_densenet_lean(config):
    device = config.device
    ft_begin_idx = config.ft_begin_idx
    sample_size = config.sample_size[0]
    sample_duration = config.sample_duration

    model = densenet88(num_classes=2,
                       sample_size=sample_size,
                       sample_duration=sample_duration).to(device)

    state_dict = torch.load(g_path +'/VioNet/'+ 'weights/DenseNetLean_Kinetics.pth')
    model.load_state_dict(state_dict)

    params = dn.get_fine_tuning_params(model, ft_begin_idx)

    return model, params
