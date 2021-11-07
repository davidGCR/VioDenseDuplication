
import add_path

import sys
# sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/VioDenseDuplication/src/VioNet')
from models.v_d_config import *

from PIL import Image, ImageFile
# from VioNet.lib.accuracy import get_accuracy
from VioNet.model_transformations import *

# from VioNet.dataset import make_dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.tensorboard import SummaryWriter
import torch

import torchvision.transforms as transforms
import os

#data
from customdatasets.make_dataset import MakeRWF2000, MakeHockeyDataset, MakeRLVDDataset, MakeUCFCrime2LocalClips
from customdatasets.make_UCFCrime import MakeUCFCrime
from customdatasets.tube_dataset import TubeDataset, my_collate, my_collate_2, OneVideoTubeDataset, TubeFeaturesDataset
from torch.utils.data import DataLoader

from config import Config
from epoch import calculate_accuracy_2
from utils import Log, save_checkpoint, load_checkpoint
from utils import get_torch_device
from torch import nn
from epoch import train, val
import numpy as np
from global_var import *
from models.mil_loss import MIL
from models.violence_detector import *

from model_transformations import *
from lib.train_script import train, val, train_regressor, train_2d_branch, val_2d_branch
from lib.train_2it_script import train_2it, val_2it
from lib.accuracy import *
from VioNet.customdatasets.vio_db import ViolenceDataset
from VioNet.transformations.temporal_transforms import RandomCrop, CenterCrop
from VioNet.train_MIL_and_2it import MIL_training

import matplotlib.pyplot as plt
# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def load_make_dataset(dataset_name, 
                    train=True, 
                    cv_split=1, 
                    home_path='', 
                    category=2, 
                    shuffle=False,
                    load_gt=False):
    if dataset_name == RWF_DATASET:
        make_dataset = MakeRWF2000(
            root=os.path.join(home_path, 'RWF-2000/frames'),
            train=train,
            category=category, 
            path_annotations=os.path.join(home_path, 'ActionTubes/final/rwf'),
            shuffle=shuffle)

    elif dataset_name == HOCKEY_DATASET:
        make_dataset = MakeHockeyDataset(
            root=os.path.join(home_path, 'HockeyFightsDATASET/frames'), 
            train=train,
            cv_split_annotation_path=os.path.join(home_path, 'VioNetDB-splits/hockey_jpg{}.json'.format(cv_split)), #'/content/DATASETS/VioNetDB-splits/hockey_jpg{}.json'
            path_annotations=os.path.join(home_path, 'ActionTubes/final/hockey'),
            )
    elif dataset_name == RLVSD_DATASET:
        make_dataset = MakeRLVDDataset(
            root=os.path.join(home_path, 'RealLifeViolenceDataset/frames'), 
            train=train,
            cv_split_annotation_path=os.path.join(home_path, 'VioNetDB-splits/RealLifeViolenceDataset{}.json'.format(cv_split)), #'/content/DATASETS/VioNetDB-splits/hockey_jpg{}.json'
            path_annotations=os.path.join(home_path, 'ActionTubes/RealLifeViolenceDataset'),
            )
    # elif dataset_name == UCFCrime_DATASET:
    #     ann_file  = ('Train_annotation.pkl', 'Train_normal_annotation.pkl') if train else ('Test_annotation.pkl', 'Test_normal_annotation.pkl')
    #     make_dataset = MakeUCFCrime(
    #         root=os.path.join(home_path, 'UCFCrime/frames'), 
    #         sp_abnormal_annotations_file=os.path.join(home_path,'VioNetDB-splits/UCFCrime', ann_file[0]), 
    #         sp_normal_annotations_file=os.path.join(home_path,'VioNetDB-splits/UCFCrime', ann_file[1]), 
    #         action_tubes_path=os.path.join(home_path,'ActionTubes/UCFCrime_reduced', ann_file[1]),
    #         train=train,
    #         ground_truth_tubes=False)
    elif dataset_name == UCFCrimeReduced_DATASET:
        ann_file  = ('Train_annotation.pkl', 'Train_normal_annotation.pkl') if train else ('Test_annotation.pkl', 'Test_normal_annotation.pkl')
        make_dataset = MakeUCFCrime(
            root=os.path.join(home_path, 'UCFCrime_Reduced', 'frames'), 
            sp_abnormal_annotations_file=os.path.join(home_path,'VioNetDB-splits/UCFCrime', ann_file[0]), 
            sp_normal_annotations_file=os.path.join(home_path,'VioNetDB-splits/UCFCrime', ann_file[1]), 
            action_tubes_path=os.path.join(home_path,'ActionTubesV2/UCFCrime_Reduced'),
            train=train,
            ground_truth_tubes=load_gt)

    return make_dataset

def main(config: Config, MIL=False):
    device = config.device
    make_dataset = load_make_dataset(
        config.dataset, 
        train=True,
        cv_split=config.num_cv,
        home_path=config.home_path,
        category=2,
        load_gt=config.load_gt)

    #validation
    val_make_dataset = load_make_dataset(
        config.dataset,
        train=False,
        cv_split=config.num_cv,
        home_path=config.home_path,
        category=2)
    # train_loader, val_loader = data_with_tubes(config, make_dataset, val_make_dataset)
    train_loader, val_loader = data_with_tubes(config, make_dataset, val_make_dataset)
   
    if config.model == 'TwoStreamVD_Binary_CFam' or config.model == 'MIL_TwoStreamVD_Binary_CFam':
        model = TwoStreamVD_Binary_CFam(config.model_config).to(device)
        if config.model_config['load_weigths'] is not None:
            print('Loading model from checkpoint...')
            model, _, _, _, _ = load_checkpoint(
                model, 
                config.device,
                None,
                config.model_config['load_weigths']
                )
        params = model.parameters()
    elif config.model == 'ResNet2D_Stream':
        model = ResNet2D_Stream(config.model_config).to(device)
        params = model.parameters()

    if MIL:
        val_make_dataset = MakeUCFCrime2LocalClips(
            root='/media/david/datos/Violence DATA/UCFCrime2LocalClips/UCFCrime2LocalClips',
            # root_normal='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/frames',
            path_annotations='/media/david/datos/Violence DATA/UCFCrime2LocalClips/Txt annotations-longVideos',
            path_person_detections='/media/david/datos/Violence DATA/PersonDetections/ucfcrime2local',
            abnormal=True)
        TWO_STREAM_INPUT_val = {
            'input_1': {
                'type': 'rgb',
                'spatial_transform': i3d_video_transf()['val'],
                'temporal_transform': CenterCrop(size=16, stride=1, input_type='rgb')
            },
            'input_2': {
                'type': 'rgb',
                'spatial_transform': resnet_transf()['val'],
                'temporal_transform': None
            }
        }
        MIL_training(config, model, train_loader, val_make_dataset, TWO_STREAM_INPUT_val)
        return 0 
    exp_config_log = config.log
    
    h_p = HOME_DRIVE if config.home_path==HOME_COLAB else config.home_path
    tsb_path_folder = os.path.join(h_p, PATH_TENSORBOARD, exp_config_log)
    chk_path_folder = os.path.join(h_p, PATH_CHECKPOINT, exp_config_log)

    for p in [tsb_path_folder, chk_path_folder]:
        if not os.path.exists(p):
            os.makedirs(p)
    # print('tensorboard dir:', tsb_path)                                                
    writer = SummaryWriter(tsb_path_folder)

    if config.optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(
            params, 
            lr=config.learning_rate, 
            eps=1e-8)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=params,
                                    lr=config.learning_rate,
                                    momentum=0.9,
                                    weight_decay=1e-3)
    elif config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            params, 
            lr=config.learning_rate, 
            eps=1e-3, 
            amsgrad=True)            
    
    if config.criterion == 'CEL':
        criterion = nn.CrossEntropyLoss().to(config.device)
    elif config.criterion == 'BCE':
        criterion = nn.BCELoss().to(config.device)
    
    
    start_epoch = 0
    ##Restore training
    if config.restore_training:
        print('Restoring training from: ', config.checkpoint_path)
        model, optimizer, epochs, last_epoch, last_loss = load_checkpoint(model, config.device, optimizer, config.checkpoint_path)
        start_epoch = last_epoch+1
        # config.num_epoch = epochs
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        verbose=True,
        factor=config.factor,
        min_lr=config.min_lr)
    
    for epoch in range(start_epoch, config.num_epoch):
        # epoch = last_epoch+i
        train_loss, train_acc = train(
            train_loader, 
            epoch, 
            model, 
            criterion, 
            optimizer, 
            config.device, 
            config, 
            calculate_accuracy_2)
        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('training accuracy', train_acc, epoch)
        
        val_loss, val_acc = val(
            val_loader,
            epoch, 
            model, 
            criterion,
            config.device,
            config,
            calculate_accuracy_2)
        scheduler.step(val_loss)
        writer.add_scalar('validation loss', val_loss, epoch)
        writer.add_scalar('validation accuracy', val_acc, epoch)

        # writer.add_scalars('loss', {'train': train_loss}, epoch)
        # writer.add_scalars('loss', {'valid': val_loss}, epoch)

        # writer.add_scalars('acc', {'train': train_acc}, epoch)
        # writer.add_scalars('acc', {'valid': val_acc}, epoch)

        if (epoch+1)%config.save_every == 0:
            save_checkpoint(model, config.num_epoch, epoch, optimizer,train_loss, os.path.join(chk_path_folder,"save_at_epoch-"+str(epoch)+".chk"))

def data_with_tubes(config: Config, make_dataset_train, make_dataset_val):
    TWO_STREAM_INPUT_train = {
        'input_1': {
            'type': 'rgb',
            # 'spatial_transform': i3d_video_transf()['train'],
            'spatial_transform': cnn3d_transf()['train'],
            'temporal_transform': None
        },
        'input_2': {
            'type': 'rgb',
            'spatial_transform': resnet_transf()['train'],
            # 'spatial_transform': cnn3d_transf()['train'],
            'temporal_transform': None
        }
        # 'input_2': {
        #     'type': 'dynamic-image',
        #     'spatial_transform': resnet_di_transf()['train'],
        #     'temporal_transform': None
        # }
    }

    TWO_STREAM_INPUT_val = {
        'input_1': {
            'type': 'rgb',
            # 'spatial_transform': i3d_video_transf()['val'],
            'spatial_transform': cnn3d_transf()['val'],
            'temporal_transform': CenterCrop(size=16, stride=1, input_type='rgb')
        },
        'input_2': {
            'type': 'rgb',
            'spatial_transform': resnet_transf()['val'],
            'temporal_transform': None
        }
        # 'input_2': {
        #     'type': 'dynamic-image',
        #     'spatial_transform': resnet_di_transf()['val'],
        #     'temporal_transform': None
        # }
    }
    train_dataset = TubeDataset(frames_per_tube=config.frames_per_tube, 
                            make_function=make_dataset_train,
                            max_num_tubes=config.num_tubes,
                            train=True,
                            dataset=config.dataset,
                            random=config.tube_sampling_random,
                            config=TWO_STREAM_INPUT_train)
    train_loader = DataLoader(train_dataset,
                        batch_size=config.train_batch,
                        shuffle=False,
                        num_workers=config.num_workers,
                        # pin_memory=True,
                        collate_fn=my_collate,
                        # sampler=train_dataset.get_sampler(),
                        drop_last=True
                        )
    val_dataset = TubeDataset(frames_per_tube=config.frames_per_tube, 
                            make_function=make_dataset_val,
                            max_num_tubes=config.num_tubes,
                            train=False,
                            dataset=config.dataset,
                            random=config.tube_sampling_random,
                            config=TWO_STREAM_INPUT_val)
    val_loader = DataLoader(val_dataset,
                        batch_size=config.val_batch,
                        # shuffle=True,
                        num_workers=config.num_workers,
                        sampler=val_dataset.get_sampler(),
                        # pin_memory=True,
                        collate_fn=my_collate,
                        drop_last=True
                        )
    return train_loader, val_loader


def data_without_tubes(config: Config, make_dataset_train, make_dataset_val):
    TWO_STREAM_INPUT_train = {
        'input_1': {
            'type': 'rgb',
            'spatial_transform': i3d_video_transf()['train'],
            'temporal_transform': RandomCrop(size=16, stride=1, input_type='rgb')
        },
        # 'input_2': {
        #     'type': 'rgb',
        #     'spatial_transform': resnet_transf()['train'],
        #     'temporal_transform': None
        # }
        'input_2': {
            'type': 'dynamic-image',
            'spatial_transform': resnet_di_transf()['train'],
            'temporal_transform': None
        }
    }

    TWO_STREAM_INPUT_val = {
        'input_1': {
            'type': 'rgb',
            'spatial_transform': i3d_video_transf()['val'],
            'temporal_transform': CenterCrop(size=16, stride=1, input_type='rgb')
        },
        'input_2': {
            'type': 'rgb',
            'spatial_transform': resnet_transf()['val'],
            'temporal_transform': None
        }
        # 'input_2': {
        #     'type': 'dynamic-image',
        #     'spatial_transform': resnet_di_transf()['val'],
        #     'temporal_transform': None
        # }
    }

    train_dataset = ViolenceDataset(
        make_dataset_train,
        dataset=config.dataset,
        config=TWO_STREAM_INPUT_train
    )

    train_loader = DataLoader(train_dataset,
                        batch_size=config.train_batch,
                        shuffle=True,
                        num_workers=config.num_workers,
                        )
    val_dataset = ViolenceDataset(
        make_dataset_val,
        dataset=config.dataset,
        config=TWO_STREAM_INPUT_val
    )

    val_loader = DataLoader(val_dataset,
                        batch_size=config.train_batch,
                        shuffle=True,
                        num_workers=config.num_workers,
                        )
    
    return train_loader, val_loader



if __name__=='__main__':
    config = Config(
        model='TwoStreamVD_Binary_CFam',#'TwoStreamVD_Binary_CFam',#'TwoStreamVD_Binary',#'i3d-roi',i3d+roi+fc
        model_config=TWO_STREAM_CFAM_CONFIG,
        # head=BINARY,
        dataset=UCFCrimeReduced_DATASET,
        num_cv=1,
        # input_type='',
        device=get_torch_device(),
        num_epoch=100,
        criterion='CEL',
        optimizer='Adadelta',
        learning_rate=0.001, #0.001 for adagrad
        train_batch=4,
        val_batch=4,
        num_tubes=4,
        tube_sampling_random=True,
        frames_per_tube=16, 
        tube_sample_strategy=MIDDLE,
        save_every=5,
        # freeze=False,
        additional_info='',
        home_path=HOME_UBUNTU,
        num_workers=4,
        load_gt=True
    )
    # config.pretrained_model = "/content/DATASETS/Pretrained_Models/DenseNetLean_Kinetics.pth"
    # config.pretrained_model='/media/david/datos/Violence DATA/VioNet_weights/pytorch_i3d/rgb_imagenet.pt'
    # config.pretrained_model = '/media/david/datos/Violence DATA/VioNet_pth/rwf_trained/save_at_epoch-127.chk'
    # config.restore_training = True
    # config.checkpoint_path = '/media/david/datos/Violence DATA/VioNet_pth/rwf-2000_model(MIL_TwoStreamVD_Binary_CFam)_config(MIL_TWO_STREAM_CFAM_CONFIG)_cv(1)_epochs(200)_num_tubes(4)_framesXtube(16)_tub_sampl_rand(True)_criterion(CEL)_optimizer(SGD)_lr(0.0001)_note(using-all-temporalinfo)/save_at_epoch-199.chk'
    # config.checkpoint_path = '/media/david/datos/Violence DATA/VioNet_pth/restoredFromDrive/save_at_epoch-49.chk'
    # config.checkpoint_path = os.path.join(config.home_path,
    #                                       PATH_CHECKPOINT,
    #                                       'rwf-2000_model(TwoStreamVD_Binary_CFam)_head(binary)_stream(rgb)_cv(1)_epochs(100)_num_tubes(4)_framesXtube(16)_tub_sampl_rand(True)_criterion(CEL)_optimizer(Adadelta)_lr(0.001)_note(TWO_STREAM_CFAM_CONFIG+finalRWF)',
    #                                       'save_at_epoch-29.chk')
    torch.autograd.set_detect_anomaly(True)
    main(config, MIL=False)
    # MIL_training(config)
    # extract_features(config, output_folder='/media/david/datos/Violence DATA/i3d-FeatureMaps/rwf')
    # load_features(config)
