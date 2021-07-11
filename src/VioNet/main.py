import os
from random import sample

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from configs_datasets import *#hockey_MDIResNet_config, hockey_i3d_config, rwf_config, environment_config, build_transforms_parameters
# from VioNet.configs_datasets import hockey_config, rwf_config
from global_var import *

from epoch import train, val, test
from model import get_model
from dataset import VioDB
from config import Config

from transformations.spatial_transforms import Compose, ToTensor, Normalize
from transformations.spatial_transforms import GroupRandomHorizontalFlip, GroupRandomScaleCenterCrop, GroupScaleCenterCrop
from transformations.temporal_transforms import CenterCrop, RandomCrop, SegmentsCrop, RandomSegmentsCrop, KeyFrameCrop, TrainGuidedKeyFrameCrop, ValGuidedKeyFrameCrop, KeySegmentCrop, IntervalCrop
from transformations.target_transforms import Label, Video

from utils import Log
from torch.utils.tensorboard import SummaryWriter

g_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(config, home_path):
    # load model
    model, params = get_model(config, home_path)
    # dataset
    dataset = config.dataset
    # sample_size = config.sample_size
    stride = config.stride
    sample_duration = config.sample_duration

    # cross validation phase
    cv = config.num_cv
    input_mode = config.input_type

    sample_size, norm = build_transforms_parameters(model_type=config.model)
    
    # train set
    crop_method = GroupRandomScaleCenterCrop(size=sample_size)
    
    # if input_mode == 'rgb':
    #     norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    # elif input_mode == 'dynamic-images':
    #     # norm = Normalize([0.49778724, 0.49780366, 0.49776983], [0.09050678, 0.09017131, 0.0898702 ])
    #     norm = Normalize([38.756858/255, 3.88248729/255, 40.02898126/255], [110.6366688/255, 103.16065604/255, 96.29023126/255])
    # else:
    #     norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_temporal_transform = build_temporal_transformation(config, config.train_temporal_transform, 'train')
    spatial_transform = Compose([crop_method,
                                 GroupRandomHorizontalFlip(),
                                 ToTensor(), 
                                 norm])
    target_transform = Label()

    train_batch = config.train_batch
    if dataset == RWF_DATASET:
        # train_data = VioDB(g_path + '/VioDB/{}_jpg/frames/'.format(dataset),
        #                 g_path + '/VioDB/{}_jpg{}.json'.format(dataset, cv), 'training',
        #                 spatial_transform, temporal_transform, target_transform, dataset,
        #                 tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path))
        train_data = VioDB(os.path.join(home_path, RWF_DATASET.upper(),'frames/'),
                            os.path.join(home_path, VIO_DB_DATASETS, "rwf-2000_jpg1.json"),
                            'training',
                            spatial_transform,
                            train_temporal_transform,
                            target_transform,
                            dataset,
                            tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path),
                            input_type=config.input_type)
    else:
        train_data = VioDB(os.path.join(home_path, VIO_DB_DATASETS, dataset+'_jpg'),#g_path + '/VioDB/{}_jpg/'.format(dataset),
                            os.path.join(home_path, VIO_DB_DATASETS,'{}_jpg{}.json'.format(dataset, cv)),#g_path + '/VioDB/{}_jpg{}.json'.format(dataset, cv),
                            'training',
                            spatial_transform,
                            train_temporal_transform,
                            target_transform,
                            dataset,
                            tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path),
                            input_type=config.input_type)
    train_loader = DataLoader(train_data,
                              batch_size=train_batch,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True)

    # val set
    crop_method = GroupScaleCenterCrop(size=sample_size)
    # if input_mode == 'rgb':
    #     norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    # elif input_mode == 'dynamic-images':
    #     norm = Normalize([0.49778724, 0.49780366, 0.49776983], [0.09050678, 0.09017131, 0.0898702 ])

    val_temporal_transform = build_temporal_transformation(config, config.val_temporal_transform, 'val')
    spatial_transform = Compose([crop_method, ToTensor(), norm])
    target_transform = Label()

    val_batch = config.val_batch

    if dataset == RWF_DATASET:
        # val_data = VioDB(g_path + '/VioDB/{}_jpg/frames/'.format(dataset),
        #                 g_path + '/VioDB/{}_jpg{}.json'.format(dataset, cv), 'validation',
        #                 spatial_transform, temporal_transform, target_transform, dataset,
        #                 tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path))
        val_data = VioDB(os.path.join(home_path, RWF_DATASET.upper(),'frames/'),
                            os.path.join(home_path, VIO_DB_DATASETS, "rwf-2000_jpg1.json"),
                            'validation',
                            spatial_transform,
                            val_temporal_transform,
                            target_transform,
                            dataset,
                            tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path),
                            input_type=config.input_type)
    else:
        val_data = VioDB(os.path.join(home_path, VIO_DB_DATASETS, dataset+'_jpg'), #g_path + '/VioDB/{}_jpg/'.format(dataset),
                        os.path.join(home_path, VIO_DB_DATASETS,'{}_jpg{}.json'.format(dataset, cv)), #g_path + '/VioDB/{}_jpg{}.json'.format(dataset, cv),
                        'validation',
                        spatial_transform,
                        val_temporal_transform, 
                        target_transform,
                        dataset,
                        tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path),
                        input_type=config.input_type)
    val_loader = DataLoader(val_data,
                            batch_size=val_batch,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    template =  '{}_fps{}_{}_split({})_input({})_TmpTransform({})_Info({})'.format(config.model,
                                                                         sample_duration,
                                                                         dataset,
                                                                         cv,
                                                                         input_mode,
                                                                         config.train_temporal_transform,
                                                                         config.additional_info
                                                                         )
    log_path = os.path.join(home_path, PATH_LOG, template)
    # chk_path = os.path.join(PATH_CHECKPOINT, template)
    tsb_path = os.path.join(home_path, PATH_TENSORBOARD, template)

    for pth in [log_path, tsb_path]:
        if not os.path.exists(pth):
            os.mkdir(pth)

    print('tensorboard dir:', tsb_path)                                                
    writer = SummaryWriter(tsb_path)

    # log
    batch_log = Log(log_path+'/batch_log.csv', ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    epoch_log = Log(log_path+'/epoch_log.csv', ['epoch', 'loss', 'acc', 'lr'])
    val_log = Log(log_path+'/val_log.csv', ['epoch', 'loss', 'acc'])
    train_val_log = Log(log_path+'/train_val_LOG.csv', ['epoch', 'train_loss', 'train_acc', 'lr', 'val_loss', 'val_acc'])

    # prepare
    criterion = nn.CrossEntropyLoss().to(config.device)
    learning_rate = config.learning_rate
    momentum = config.momentum
    weight_decay = config.weight_decay

    optimizer = torch.optim.SGD(params=params,
                                lr=learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           verbose=True,
                                                           factor=config.factor,
                                                           min_lr=config.min_lr)

    acc_baseline = config.acc_baseline
    loss_baseline = 1

    for i in range(config.num_epoch):
        train_loss, train_acc, lr = train(i, train_loader, model, criterion, optimizer, config.device, batch_log, epoch_log)
        val_loss, val_acc = val(i, val_loader, model, criterion, config.device, val_log)
        epoch = i+1
        train_val_log.log({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'lr': lr, 'val_loss': val_loss, 'val_acc': val_acc})
        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('training accuracy', train_acc, epoch)
        writer.add_scalar('validation loss', val_loss, epoch)
        writer.add_scalar('validation accuracy', val_acc, epoch)

        scheduler.step(val_loss)
        if val_acc > acc_baseline or (val_acc >= acc_baseline and val_loss < loss_baseline):
            torch.save(model.state_dict(),
                       os.path.join(home_path, PATH_CHECKPOINT,'{}_fps{}_{}{}_{}_{:.4f}_{:.6f}.pth'.format(config.model,
                                                                                                            sample_duration,
                                                                                                            dataset,
                                                                                                            cv,
                                                                                                            epoch,
                                                                                                            val_acc,
                                                                                                            val_loss)
                                    )
                        )
            acc_baseline = val_acc
            loss_baseline = val_loss

from global_var import *
# from configs_datasets import *

if __name__ == '__main__':
    
    # config = rwf_config()
    # config = hockey_MDIResNet_config()
    config = rwf_MDIResNet_config()

    if config.dataset == RWF_DATASET:
        config.num_cv = 1
        
        main(config, environment_config['home'])
    elif config.dataset == HOCKEY_DATASET:
        
        for i in range(1,2):
            config.num_cv = i
            main(config, environment_config['home'])