from re import T
from transformations.spatial_transforms import Compose, Normalize, GroupRandomScaleCenterCrop, GroupRandomHorizontalFlip, ToTensor
from transformations.temporal_transforms import *
import torchvision.transforms as transforms
from transformations.data_aug.data_aug import *

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

def cnn3d_transf():
    sample_size = (224,224)
    
    T = {
        'train':Compose(
                [
                    ClipRandomHorizontalFlip(), 
                    # ClipRandomScale(scale=0.2, diff=True), 
                    ClipRandomRotate(angle=5),
                    # ClipRandomTranslate(translate=0.1, diff=True),
                    # ClipRandomScale(scale=0.2, diff=True),
                    NumpyToTensor()
                ],
                probs=[0.4, 0.5, 0.2, 0.3]
                ),
        'val': Compose(
                [
                    NumpyToTensor()
                ]
                )
    }
    return T


def i3d_transf():
    sample_size = (224,224)
    norm = Normalize([38.756858/255, 3.88248729/255, 40.02898126/255], [110.6366688/255, 103.16065604/255, 96.29023126/255])
    T = {
        'train':transforms.Compose([
                            # transforms.CenterCrop(sample_size),
                            transforms.Resize(sample_size),
                            # transforms.RandomHorizontalFlip(0.5),
                            transforms.ToTensor(),
                            norm
                            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]),
        'val': transforms.Compose([
                            # transforms.CenterCrop(sample_size),
                            transforms.Resize(sample_size),
                            transforms.ToTensor(),
                            norm
                            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
    }
    return T

def i3d_video_transf():
    sample_size = (224,224)
    norm = Normalize([38.756858/255, 3.88248729/255, 40.02898126/255], [110.6366688/255, 103.16065604/255, 96.29023126/255])
    T = {
        'train':Compose([
                            GroupRandomScaleCenterCrop(size=sample_size),
                            GroupRandomHorizontalFlip(),
                            ToTensor(),
                            # norm
                            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]),
        'val': Compose([
                            GroupRandomScaleCenterCrop(size=sample_size),
                            ToTensor(),
                            # norm
                        ])
    }
    return T

def resnet_transf():
    # mean = [0.3833, 0.3768, 0.3709]
    # std = [0.2596, 0.2587, 0.2598]

    # old
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    input_size = (224,224)
    T = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]),
            'val': transforms.Compose([
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean=mean, std=std)
            ]),
        }
    return T  

def resnet_di_transf():
    # mean = [0.5002, 0.5001, 0.5001]
    # std = [0.0799, 0.0795, 0.0796]

    #old
    # mean = [0.49778724, 0.49780366, 0.49776983]
    # std = [0.09050678, 0.09017131, 0.0898702 ]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm = transforms.Normalize(mean=mean, std=std)  
    input_size = (224,224)
    T = {
            'train': transforms.Compose([
                # transforms.RandomResizedCrop(input_size),
                # transforms.RandomHorizontalFlip(),
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                norm
            ]),
            'val': transforms.Compose([
                # transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                norm
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
            # norm = dyn_img_transf_parameters()
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

