import os
import sys
from google.protobuf.reflection import ParseMessage
# g_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(1, g_path)

import torch
import torch.nn as nn
from torch.serialization import load

import models.densenet as dn

from models.c3d import C3D
from models.c3d_fe import C3D_FE
from models.s3d import S3D, S3D_feature_extractor
from models.densenet import densenet88, densenet121
from models.dense_net_roi import densenet88_roi
from models.convlstm import ConvLSTM
from models.models2D import ResNet, Densenet2D, FusedResNextTempPool, FeatureExtractorResNextTempPool
from models.anomaly_detector import AnomalyDetector
from models.i3d import InceptionI3d, TwoStreamI3D
from models.i3d_roi import InceptionI3d_Roi
import models.models2D as rn
# from models.violence_detector import ViolenceDetector
from global_var import *

# g_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def get_model(config, home_path):
    if config.model == 'c3d':
        model, params = VioNet_C3D(config, home_path)
    elif config.model == 'convlstm':
        model, params = VioNet_ConvLSTM(config)
    elif config.model == 'densenet':
        model, params = VioNet_densenet(config, home_path)
    elif config.model == 'densenet_lean':
        model, params = VioNet_densenet_lean(config, home_path)
    elif config.model == 'densenet_lean_roi':
        model, params = VioNet_densenet_lean_roi(config, home_path)
    elif config.model == 'resnet50':
        model, params = VioNet_Resnet(config, home_path)
    elif config.model == 'densenet2D':
        model, params = VioNet_Densenet2D(config)
    elif config.model == 'i3d':
        model, params = VioNet_I3D(config)
    elif config.model == 'i3d-roi':
        model, params = VioNet_I3D_Roi(config)
    elif config.model == 'two-i3d':
        model, params = VioNet_TwoStreamI3D(config)
    elif config.model == 'two-i3dv2':
        model, params = VioNet_TwoStreamI3D(config, freeze=True)
    elif config.model == 's3d':
        model, params = VioNet_S3D(config)
    elif config.model == 'MDIResNet':
        model, params = MDI_ResNet(config)
    else:
        model, params = VioNet_densenet_lean(config, home_path)
    
    return model, params

def get_two_models(config, pretrained_path_1, pretrained_path_2):
    if config.model == 'i3d':
        config.pretrained_model = pretrained_path_1
        model_1, _ = VioNet_I3D(config)
        config.pretrained_model = pretrained_path_2
        model_2, _ = VioNet_I3D(config)
    return model_1, model_2

def MDI_ResNet(config):
    model = ResNet(num_classes=2, model_name='resnet50').to(config.device)
    params = model.parameters()
    return model, params

from utils import load_checkpoint

# def ViolenceDetector_model(config, device, pretrained_model=None):
#     #with default config
#     model = ViolenceDetector(classifier=config.head,
#                             freeze=config.freeze).to(device)
#     if pretrained_model:
#         # if device == torch.device('cpu'):
#         #     checkpoint = torch.load(config.pretrained_model, map_location=device)    
#         # else:
#         #     checkpoint = torch.load(config.pretrained_model)
#         # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
#         model, _, _, _, _ = load_checkpoint(model, device, None, pretrained_model)
#     params = model.parameters()
#     return model, params


def Feature_Extractor_C3D(device, pretrained_model):
    if pretrained_model:
        model = C3D_FE(pretrained=pretrained_model).to(device)
    else:
        model = C3D_FE(pretrained=False).to(device)
    return model
    

def Feature_Extractor_S3D(device, pretrained_model):
    model = S3D_feature_extractor().to(device)
    if pretrained_model:
        if device == torch.device('cpu'):
            weight_dict = torch.load(pretrained_model, map_location=device)
        else:
            weight_dict = torch.load(pretrained_model)
        
        model.load_state_dict(weight_dict, strict=False)

        # model_dict = model.state_dict()
        # for name, param in weight_dict.items():
        #     if 'module' in name:
        #         name = '.'.join(name.split('.')[1:])
        #     if name in model_dict:
        #         if param.size() == model_dict[name].size():
        #             model_dict[name].copy_(param)
        #         else:
        #             print (' size? ' + name, param.size(), model_dict[name].size())
        #     else:
        #         print (' name? ' + name)
    return model

def AnomalyDetector_model(config, source):

    device = config.device
    if source==FEAT_EXT_RESNEXT or source == FEAT_EXT_C3D:
        model = AnomalyDetector(config.input_dimension).to(device)
    elif source==FEAT_EXT_RESNEXT_S3D:
        model = AnomalyDetector(config.input_dimension[0]+config.input_dimension[1]).to(device)

    # if config.pretrained_model:
    #     if device == torch.device('cpu'):
    #         state_dict = torch.load(config.pretrained_model, map_location=device)    
    #     else:
    #         state_dict = torch.load(config.pretrained_model)
    #     model.load_state_dict(state_dict, strict=False)

    if config.pretrained_model:
        if device == torch.device('cpu'):
            checkpoint = torch.load(config.pretrained_model, map_location=device)    
        else:
            checkpoint = torch.load(config.pretrained_model)
        # if not source == FEAT_EXT_C3D:
        #     model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # else:
        #     model.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    params = model.parameters()
    return model, params


def VioNet_Densenet2D(config):
    device = config.device
    model = Densenet2D(num_classes=2).to(device)
    params = rn.get_fine_tuning_params(model, config.ft_begin_idx)
    return model, params

def VioNet_Resnet(config, home_path):
    device = config.device
    model = ResNet(num_classes=config.num_classes).to(device)
    if config.pretrained_model:
        # state_dict = torch.load(g_path +'/VioNet/weights/'+ config.pretrained_model)
        state_dict = torch.load(os.path.join(home_path, VIONET_WEIGHTS, config.pretrained_model))
        model.load_state_dict(state_dict)

    params = rn.get_fine_tuning_params(model, config.ft_begin_idx)
    return model, params

def VioNet_ResnetXT(config):
    device = config.device
    model = FusedResNextTempPool(num_classes=config.num_classes).to(device)
    if config.pretrained_model:
        state_dict = torch.load(config.pretrained_model)
        model.load_state_dict(state_dict)

    # params = rn.get_fine_tuning_params(model, config.ft_begin_idx)
    params = model.parameters()
    return model, params

def FeatureExtractor_ResnetXT(device, pretrained_model):
    
    model = FeatureExtractorResNextTempPool().to(device)
    if pretrained_model:
        if device == torch.device('cpu'):
            state_dict = torch.load(pretrained_model, map_location=device)    
        else:
            state_dict = torch.load(pretrained_model)
        model.load_state_dict(state_dict, strict=False)

    # params = rn.get_fine_tuning_params(model, config.ft_begin_idx)
    return model

def VioNet_C3D(config, home_path):
    device = config.device
    model = C3D(num_classes=2).to(device)

    # print(model)

    # state_dict = torch.load(g_path +'/VioNet/'+ 'weights/C3D_Kinetics.pth')
    state_dict = torch.load(os.path.join(home_path, VIONET_WEIGHTS, 'C3D_Kinetics.pth'))
    
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

def VioNet_I3D_Roi(config, device, pretrained_model):
    """
    Load I3D model
        config.device
        config.pretrained_model
    """
    model = InceptionI3d_Roi(num_classes=400, in_channels=3).to(device)
    if pretrained_model:
        state_dict = torch.load(pretrained_model)
        model.load_state_dict(state_dict,  strict=False)
        model.replace_logits(2)
    model.to(device)
    params = model.parameters()
    return model, params

def VioNet_I3D(config):
    """
    Load I3D model
        config.device
        config.pretrained_model
    """
    model = InceptionI3d(num_classes=400, in_channels=3).to(config.device)
    if  not config.pretrained_model:
        config.pretrained_model = '/media/david/datos/Violence DATA/VioNet_weights/pytorch_i3d/rgb_imagenet.pt' 
        # model = nn.DataParallel(model)
        state_dict = torch.load(config.pretrained_model)
        model.load_state_dict(state_dict,  strict=False)
        model.replace_logits(2)
    else:
        model.replace_logits(2)
        state_dict = torch.load(config.pretrained_model)
        model.load_state_dict(state_dict,  strict=False)
    model.to(config.device)
    params = model.parameters()
    return model, params

def VioNet_TwoStreamI3D(config, freeze=False):
    if  not config.pretrained_model:
        config.pretrained_model = '/media/david/datos/Violence DATA/VioNet_weights/pytorch_i3d/rgb_imagenet.pt' 
    
    model = TwoStreamI3D(num_classes=2, num_features=2048, load_model_path=config.pretrained_model, freeze=freeze).to(config.device)
    params = model.parameters()
    return model, params

def VioNet_S3D(config):
    """
    Load S3D model
        config.device
        config.pretrained_model
    """
    model = S3D(num_class=2).to(config.device)
    if not config.pretrained_model:
        config.pretrained_model = '/media/david/datos/Violence DATA/VioNet_weights/S3D_kinetics400.pt'
    state_dict = torch.load(config.pretrained_model)
    model.load_state_dict(state_dict,  strict=False)

    params = model.parameters()
    return model, params

def VioNet_densenet(config, home_path):
    """
    Load DENSENET model
        config.device
        config.pretrained_model
        config.sample_size
        config.sample_duration
    """
    device = config.device
    ft_begin_idx = config.ft_begin_idx
    sample_size = config.sample_size[0]
    sample_duration = config.sample_duration

    model = densenet121(num_classes=2,
                        sample_size=sample_size,
                        sample_duration=sample_duration).to(device)

    # state_dict = torch.load(g_path +'/VioNet/'+ 'weights/DenseNet_Kinetics.pth')
    state_dict = torch.load(os.path.join(home_path, VIONET_WEIGHTS, 'DenseNet_Kinetics.pth'))
    model.load_state_dict(state_dict)

    params = dn.get_fine_tuning_params(model, ft_begin_idx)

    return model, params


# the model we finally adopted in DenseNet
def VioNet_densenet_lean(config, home_path):
    device = config.device
    ft_begin_idx = config.ft_begin_idx
    sample_size = config.sample_size[0]
    sample_duration = config.sample_duration

    model = densenet88(num_classes=2,
                       sample_size=sample_size,
                       sample_duration=sample_duration).to(device)

    # state_dict = torch.load(g_path +'/VioNet/'+ 'weights/DenseNetLean_Kinetics.pth')
    state_dict = torch.load(os.path.join(home_path, VIONET_WEIGHTS, 'DenseNetLean_Kinetics.pth'))
    model.load_state_dict(state_dict)

    params = dn.get_fine_tuning_params(model, ft_begin_idx)

    return model, params

def VioNet_densenet_lean_roi(config, pretrained):
    device = config.device
    ft_begin_idx = config.ft_begin_idx
    sample_size = config.sample_size[0]
    sample_duration = config.sample_duration

    model = densenet88_roi(num_classes=2,
                       sample_size=sample_size,
                       sample_duration=sample_duration).to(device)

    # state_dict = torch.load(g_path +'/VioNet/'+ 'weights/DenseNetLean_Kinetics.pth')
    state_dict = torch.load(pretrained)
    

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # if k[0:2] == 'fc':
        #     continue
        name = k[9:] # remove 'module.' of dataparallel
        # print(k, '--', name)
        new_state_dict[name]=v

    
    model.load_state_dict(new_state_dict, strict=False)

    params = dn.get_fine_tuning_params(model, ft_begin_idx)

    return model, params


if __name__=="__main__":
    print("Hey there!!!: ")