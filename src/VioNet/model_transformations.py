from re import T
from transformations.spatial_transforms import Normalize, GroupRandomHorizontalFlip
from transformations.temporal_transforms import *
import torchvision.transforms as transforms

i3d_based_models = [
    'TwoStreamVD_Binary_CFam',
    'TwoStreamVD_Binary',
    'i3d',
    'two-i3d',
    'two-i3dv2'
]

models_2d_ = [
        'TwoStreamVD_Binary',
        'TwoStreamVD_Binary_CFam'
    ]

def i3d_transf():
    sample_size = (224,224)
    norm = Normalize([38.756858/255, 3.88248729/255, 40.02898126/255], [110.6366688/255, 103.16065604/255, 96.29023126/255])
    T = {
        'train':transforms.Compose([
                            transforms.CenterCrop(sample_size),
                            # transforms.Resize(sample_size),
                            # transforms.RandomHorizontalFlip(0.5),
                            transforms.ToTensor(),
                            norm
                            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]),
        'val': transforms.Compose([
                            transforms.CenterCrop(sample_size),
                            # transforms.Resize(sample_size),
                            transforms.ToTensor(),
                            norm
                            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
    }
    return T

def resnet_transf():
    input_size = (224,224)
    T = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    return T     

class DefaultTrasformations:
    def __init__(self, model_name, size=None, mean=None, std=None, train=True):
        self.model_name = model_name
        self.size = size
        self.mean = mean
        self.std = std
        self.train = train
    

    
    def __preprocess__(self):
        sample_size, norm = None, None
        if self.model_name in i3d_based_models:
            sample_size = (224,224) if not self.size else self.size
            norm = Normalize([38.756858/255, 3.88248729/255, 40.02898126/255], [110.6366688/255, 103.16065604/255, 96.29023126/255])
        elif self.model_name == 's3d':
            sample_size = (224,224) if not self.size else self.size
            norm = Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) #from pytorch
        elif self.model_name == 'densenet_lean':
            sample_size = (112,112) if not self.size else self.size
            norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        elif self.model_name == 'densenet_lean_roi':
            sample_size = (112,112) if not self.size else self.size
            norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        elif self.model_name == 'MDIResNet':
            sample_size = (224,224) if not self.size else self.size
            norm = dyn_img_transf_parameters()
        else:
            print('Loading default spatial transformations')
            sample_size = (224,224) if not self.size else self.size
            norm = Normalize([38.756858/255, 3.88248729/255, 40.02898126/255], [110.6366688/255, 103.16065604/255, 96.29023126/255])

        return sample_size, norm

    # def __call__(self):
    #     sample_size, norm = self.__preprocess__()
    #     if self.train:
    #         return transforms.Compose([
    #                             # transforms.CenterCrop(224),
    #                             transforms.Resize(sample_size),
    #                             transforms.ToTensor(),
    #                             norm
    #                             # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #                         ])
    #     else:
    #         return transforms.Compose([
    #                             # transforms.CenterCrop(224),
    #                             transforms.Resize(sample_size),
    #                             transforms.ToTensor(),
    #                             norm
    #                             # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #                         ])

        


# def build_transforms_parameters(model_type):
#     if model_type in i3d_based_models:
#         sample_size = (224,224)
#         norm = Normalize([38.756858/255, 3.88248729/255, 40.02898126/255], [110.6366688/255, 103.16065604/255, 96.29023126/255])
#     elif model_type == 's3d':
#         sample_size = (224,224)
#         norm = Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) #from pytorch
#     elif model_type == 'densenet_lean':
#         sample_size = (112,112)
#         norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     elif model_type == 'MDIResNet':
#         sample_size = (224,224)
#         norm = dyn_img_transf_parameters()
#     return sample_size, norm

def dyn_img_transf_parameters():
    return Normalize([0.49778724, 0.49780366, 0.49776983], [0.09050678, 0.09017131, 0.0898702 ])