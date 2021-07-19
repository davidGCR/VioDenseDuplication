import os
from random import sample

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from global_var import *

from epoch import train, val, test, calculate_accuracy_2
from model import get_model, get_two_models
from dataset import VioDB
from config import Config

from transformations.spatial_transforms import Compose, ToTensor, Normalize
from transformations.spatial_transforms import GroupRandomHorizontalFlip, GroupRandomScaleCenterCrop, GroupScaleCenterCrop
from transformations.temporal_transforms import CenterCrop, RandomCrop, SegmentsCrop, RandomSegmentsCrop, KeyFrameCrop, TrainGuidedKeyFrameCrop, ValGuidedKeyFrameCrop, KeySegmentCrop, IntervalCrop
from transformations.target_transforms import Label, Video

from utils import Log
from torch.utils.tensorboard import SummaryWriter

from configs_datasets import *

# g_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def build_temporal_transformation(config: Config, transf_type: str):
    """
    config.sample_duration
    config.stride
    config.input_type
    conig.segment_size
    """
    if transf_type == STANDAR_CROP:
        temporal_transform = RandomCrop(size=config.sample_duration, stride=config.stride, input_type=config.input_type)
    elif transf_type == SEGMENTS_CROP:
        temporal_transform = SegmentsCrop(size=config.sample_duration, segment_size=config.segment_size, stride=config.stride, overlap=config.overlap)
    elif transf_type == CENTER_CROP:
        temporal_transform = CenterCrop(size=config.sample_duration, stride=config.stride, input_type=config.input_type)
    elif transf_type == KEYFRAME_CROP:
        temporal_transform = KeyFrameCrop(size=config.sample_duration, stride=config.stride, input_type=config.input_type)
    elif transf_type == GUIDED_KEYFRAME_CROP:
        temporal_transform = TrainGuidedKeyFrameCrop(size=config.sample_duration, segment_size=config.segment_size, stride=config.stride, overlap=0.5)
    elif transf_type == KEYSEGMENT_CROP:
        temporal_transform = KeySegmentCrop(size=config.sample_duration, stride=config.stride, input_type=config.input_type, segment_type="highestscore")
    elif transf_type == INTERVAL_CROP:
        temporal_transform = IntervalCrop(intervals_num=config.sample_duration, interval_len=config.segment_size, overlap=config.overlap)
    return temporal_transform
    

def train_two_stream(config: Config, home_path):
    model, params = get_model(config, home_path)
    # dataset
    dataset = config.dataset

    # cross validation phase
    cv = config.num_cv
    sample_size, norm = build_transforms_parameters(model_type=config.model)
    
    # val set
    crop_method = GroupScaleCenterCrop(size=sample_size)
    # RGB stream
    config.val_temporal_transform = INTERVAL_CROP
    input_type_1 = 'rgb'
    train_temporal_transform_1 = build_temporal_transformation(config, config.val_temporal_transform)
    
    # DI stream
    config.val_temporal_transform = SEGMENTS_CROP
    input_type_2 = 'dynamic-images'
    train_temporal_transform_2 = build_temporal_transformation(config, config.val_temporal_transform)
    
    
    spatial_transform = Compose([crop_method, ToTensor(), norm])
    target_transform = Label()
    train_batch = config.train_batch

    if dataset == RWF_DATASET:
        train_data_1 = VioDB(os.path.join(home_path, RWF_DATASET.upper(),'frames/'),
                            os.path.join(home_path, VIO_DB_DATASETS, "rwf-2000_jpg1.json"),
                            'training',
                            spatial_transform,
                            train_temporal_transform_1,
                            target_transform,
                            dataset,
                            tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path),
                            input_type=input_type_1)
        train_data_2 = VioDB(os.path.join(home_path, RWF_DATASET.upper(),'frames/'),
                            os.path.join(home_path, VIO_DB_DATASETS, "rwf-2000_jpg1.json"),
                            'training',
                            spatial_transform,
                            train_temporal_transform_2,
                            target_transform,
                            dataset,
                            tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path),
                            input_type=input_type_2)
        val_data_1 = VioDB(os.path.join(home_path, RWF_DATASET.upper(),'frames/'),
                            os.path.join(home_path, VIO_DB_DATASETS, "rwf-2000_jpg1.json"),
                            'validation',
                            spatial_transform,
                            train_temporal_transform_1,
                            target_transform,
                            dataset,
                            tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path),
                            input_type=input_type_1)
        val_data_2 = VioDB(os.path.join(home_path, RWF_DATASET.upper(),'frames/'),
                            os.path.join(home_path, VIO_DB_DATASETS, "rwf-2000_jpg1.json"),
                            'validation',
                            spatial_transform,
                            train_temporal_transform_2,
                            target_transform,
                            dataset,
                            tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path),
                            input_type=input_type_2)
    else:
        train_data_1 = VioDB(os.path.join(home_path, VIO_DB_DATASETS, dataset+'_jpg'), #g_path + '/VioDB/{}_jpg/'.format(dataset),
                        os.path.join(home_path, VIO_DB_DATASETS,'{}_jpg{}.json'.format(dataset, cv)), #g_path + '/VioDB/{}_jpg{}.json'.format(dataset, cv),
                        'training',
                        spatial_transform,
                        train_temporal_transform_1, 
                        target_transform,
                        dataset,
                        tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path),
                        input_type=input_type_1)
        train_data_2 = VioDB(os.path.join(home_path, VIO_DB_DATASETS, dataset+'_jpg'), #g_path + '/VioDB/{}_jpg/'.format(dataset),
                        os.path.join(home_path, VIO_DB_DATASETS,'{}_jpg{}.json'.format(dataset, cv)), #g_path + '/VioDB/{}_jpg{}.json'.format(dataset, cv),
                        'training',
                        spatial_transform,
                        train_temporal_transform_2, 
                        target_transform,
                        dataset,
                        tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path),
                        input_type=input_type_2)
        val_data_1 = VioDB(os.path.join(home_path, VIO_DB_DATASETS, dataset+'_jpg'), #g_path + '/VioDB/{}_jpg/'.format(dataset),
                        os.path.join(home_path, VIO_DB_DATASETS,'{}_jpg{}.json'.format(dataset, cv)), #g_path + '/VioDB/{}_jpg{}.json'.format(dataset, cv),
                        'validation',
                        spatial_transform,
                        train_temporal_transform_1, 
                        target_transform,
                        dataset,
                        tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path),
                        input_type=input_type_1)
        val_data_2 = VioDB(os.path.join(home_path, VIO_DB_DATASETS, dataset+'_jpg'), #g_path + '/VioDB/{}_jpg/'.format(dataset),
                        os.path.join(home_path, VIO_DB_DATASETS,'{}_jpg{}.json'.format(dataset, cv)), #g_path + '/VioDB/{}_jpg{}.json'.format(dataset, cv),
                        'validation',
                        spatial_transform,
                        train_temporal_transform_2, 
                        target_transform,
                        dataset,
                        tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path),
                        input_type=input_type_2)

    train_loader_1 = DataLoader(train_data_1,
                            batch_size=train_batch,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True)
    train_loader_2 = DataLoader(train_data_2,
                            batch_size=train_batch,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True)
    val_batch = train_batch
    val_loader_1 = DataLoader(val_data_1,
                            batch_size=val_batch,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
    val_loader_2 = DataLoader(val_data_2,
                            batch_size=val_batch,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    
    template =  '{}_fps{}_{}_split({})_input({})_TmpTransform({})_Info({})'.format(config.model,
                                                                         config.sample_duration,
                                                                         dataset,
                                                                         cv,
                                                                         input_type_1+'+'+input_type_2,
                                                                         INTERVAL_CROP+'+'+SEGMENTS_CROP,
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
    # batch_log = Log(log_path+'/batch_log.csv', ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    # epoch_log = Log(log_path+'/epoch_log.csv', ['epoch', 'loss', 'acc', 'lr'])

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
    from utils import AverageMeter

    stages = ['train', 'val']

    for epoch in range(config.num_epoch):
        for stage in stages:
            # meters
            losses = AverageMeter()
            accuracies = AverageMeter()
            if stage == 'train':
                loader_1, loader_2 = train_loader_1, train_loader_2
                print('training at epoch: {}'.format(epoch))
                # set model to training mode
                model.train()
            else:
                loader_1, loader_2 = val_loader_1, val_loader_2
                print('validation at epoch: {}'.format(epoch))
                model.eval()

            for i, (data_1, data_2) in enumerate(zip(loader_1, loader_2)):
                inputs_1, targets_1 = data_1
                inputs_2, targets_2 = data_2
                inputs_1, targets_1 = inputs_1.to(device), targets_1.to(device)
                inputs_2, targets_2 = inputs_2.to(device), targets_2.to(device)

                if stage == 'train':
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model((inputs_1, inputs_2))
                    loss = criterion(outputs, targets_1)
                    acc = calculate_accuracy_2(outputs, targets_1)
                else:
                    # no need to track grad in eval mode
                    with torch.no_grad():
                        outputs = model((inputs_1, inputs_2))
                        loss = criterion(outputs, targets_1)
                        acc = calculate_accuracy_2(outputs, targets_2)


                # meter
                losses.update(loss.item(), inputs_1.size(0))
                accuracies.update(acc, inputs_1.size(0))

                if stage == 'train':
                    # backward + optimize
                    loss.backward()
                    optimizer.step()

                    print(
                        'Epoch: [{0}][{1}/{2}]\t'
                        # 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        # 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                            epoch,
                            i + 1,
                            len(train_loader_1),
                            # batch_time=batch_time,
                            # data_time=data_time,
                            loss=losses,
                            acc=accuracies
                        )
                    )
                # else:
                #     print(
                #         'Epoch: [{}]\t'
                #         'Loss(val): {loss.avg:.4f}\t'
                #         'Acc(val): {acc.avg:.3f}'.format(epoch, loss=losses, acc=accuracies)
                #     )

            if stage == 'train':
                train_loss, train_acc = losses.avg, accuracies.avg
                
                writer.add_scalar('training loss', train_loss, epoch)
                writer.add_scalar('training accuracy', train_acc, epoch)
            else:
                print(
                        'Epoch: [{}]\t'
                        'Loss(val): {loss.avg:.4f}\t'
                        'Acc(val): {acc.avg:.3f}'.format(epoch, loss=losses, acc=accuracies)
                    )
                val_loss, val_acc = losses.avg, accuracies.avg
                writer.add_scalar('validation loss', val_loss, epoch)
                writer.add_scalar('validation accuracy', val_acc, epoch)

                scheduler.step(val_loss)
                if val_acc > acc_baseline or (val_acc >= acc_baseline and val_loss < loss_baseline):
                    torch.save(model.state_dict(),
                            os.path.join(home_path, PATH_CHECKPOINT,'{}_fps{}_{}_({})_fold({})_{}_{:.4f}_{:.6f}.pth'.format(config.model,
                                                                                                                    config.sample_duration,
                                                                                                                    dataset,
                                                                                                                    'rgb+di',
                                                                                                                    cv,
                                                                                                                    epoch,
                                                                                                                    val_acc,
                                                                                                                    val_loss)
                                            )
                                )
                    acc_baseline = val_acc
                    loss_baseline = val_loss



def val_two_stream(config: Config, home_path, pretrained_model_rgb, pretrained_model_di):
    # load model
    # model_1, model_2 = get_two_models('/media/david/datos/Violence DATA/VioNet_pth/i3d_fps16_rwf-20001_15_0.8356_0.394731.pth',
    #                                   '/media/david/datos/Violence DATA/VioNet_pth/i3d_fps16_rwf_2000_(dynamic-images)_1_15_0.8044_0.426578.pth')
    model_1, model_2 = get_two_models(pretrained_model_rgb,
                                      pretrained_model_di)

    # dataset
    dataset = config.dataset

    # cross validation phase
    cv = config.num_cv
    sample_size, norm = build_transforms_parameters(model_type=config.model)
    
    # val set
    crop_method = GroupScaleCenterCrop(size=sample_size)
    # RGB stream
    config.val_temporal_transform = INTERVAL_CROP
    input_type_1 = 'rgb'
    val_temporal_transform_1 = build_temporal_transformation(config, config.val_temporal_transform)
    
    # DI stream
    config.val_temporal_transform = SEGMENTS_CROP
    input_type_2 = 'dynamic-images'
    val_temporal_transform_2 = build_temporal_transformation(config, config.val_temporal_transform)
    
    
    spatial_transform = Compose([crop_method, ToTensor(), norm])
    target_transform = Label()
    val_batch = config.val_batch

    if dataset == RWF_DATASET:
        val_data_1 = VioDB(os.path.join(home_path, RWF_DATASET.upper(),'frames/'),
                            os.path.join(home_path, VIO_DB_DATASETS, "rwf-2000_jpg1.json"),
                            'validation',
                            spatial_transform,
                            val_temporal_transform_1,
                            target_transform,
                            dataset,
                            tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path),
                            input_type=input_type_1)
        val_data_2 = VioDB(os.path.join(home_path, RWF_DATASET.upper(),'frames/'),
                            os.path.join(home_path, VIO_DB_DATASETS, "rwf-2000_jpg1.json"),
                            'validation',
                            spatial_transform,
                            val_temporal_transform_2,
                            target_transform,
                            dataset,
                            tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path),
                            input_type=input_type_2)
    else:
        val_data_1 = VioDB(os.path.join(home_path, VIO_DB_DATASETS, dataset+'_jpg'), #g_path + '/VioDB/{}_jpg/'.format(dataset),
                        os.path.join(home_path, VIO_DB_DATASETS,'{}_jpg{}.json'.format(dataset, cv)), #g_path + '/VioDB/{}_jpg{}.json'.format(dataset, cv),
                        'validation',
                        spatial_transform,
                        val_temporal_transform_1, 
                        target_transform,
                        dataset,
                        tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path),
                        input_type=input_type_1)
        val_data_2 = VioDB(os.path.join(home_path, VIO_DB_DATASETS, dataset+'_jpg'), #g_path + '/VioDB/{}_jpg/'.format(dataset),
                        os.path.join(home_path, VIO_DB_DATASETS,'{}_jpg{}.json'.format(dataset, cv)), #g_path + '/VioDB/{}_jpg{}.json'.format(dataset, cv),
                        'validation',
                        spatial_transform,
                        val_temporal_transform_2, 
                        target_transform,
                        dataset,
                        tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path),
                        input_type=input_type_2)
    val_loader_1 = DataLoader(val_data_1,
                            batch_size=val_batch,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
    val_loader_2 = DataLoader(val_data_2,
                            batch_size=val_batch,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    # prepare
    criterion = nn.CrossEntropyLoss().to(device)
    
    # set model to evaluate mode
    model_1.eval()
    model_2.eval()

    from utils import AverageMeter
    import torch.nn.functional as F
    # meters
    losses = AverageMeter()
    accuracies = AverageMeter()

    for data_1, data_2 in zip(val_loader_1, val_loader_2):
        inputs_1, targets_1 = data_1
        inputs_2, targets_2 = data_2
        inputs_1, targets_1 = inputs_1.to(device), targets_1.to(device)
        inputs_2, targets_2 = inputs_2.to(device), targets_2.to(device)

        # no need to track grad in eval mode
        with torch.no_grad():
            outputs_1 = model_1(inputs_1)
            probs_1 = F.softmax(outputs_1, dim=1)
            # print('outputs_1:', outputs_1)
            # print('probs_1:', probs_1)

            # loss_1 = criterion(outputs_1, targets_1)
            # acc_1 = calculate_accuracy_2(outputs_1, targets_1)

            outputs_2 = model_2(inputs_2)
            probs_2 = F.softmax(outputs_2, dim=1)
            # print('outputs_2:', outputs_2)
            # print('probs_2:', probs_2)

            # values, indices = torch.max(probs_1 + probs_2, 1)
            values, indices = torch.max(outputs_1 + outputs_2, 1)
            #accuracy
            acc = (indices == targets_1).sum().item() / len(indices)

            # loss_2 = criterion(outputs_2, targets_2)
            # acc_2 = calculate_accuracy_2(outputs_2, targets_2)

        # losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs_1.size(0))

    print(
        'Epoch: [{}]\t'
        'Loss(val): {loss.avg:.4f}\t'
        'Acc(val): {acc.avg:.3f}'.format(1, loss=losses, acc=accuracies)
    )
    # val_loss, val_acc = val(1, val_loader, model, criterion, device, val_log=None)
    # print("Validation loss: {}, Accuray: {}".format(val_loss, val_acc))

def val_one_stream(config, home_path):
    # load model
    model, params = get_model(config, home_path)

    # dataset
    dataset = config.dataset
    stride = config.stride
    sample_duration = config.sample_duration

    # cross validation phase
    cv = config.num_cv
    input_mode = config.input_type
    sample_size, norm = build_transforms_parameters(model_type=config.model)
    
    # val set
    crop_method = GroupScaleCenterCrop(size=sample_size)
    val_temporal_transform = build_temporal_transformation(config, config.val_temporal_transform)
    spatial_transform = Compose([crop_method, ToTensor(), norm])
    target_transform = Label()
    val_batch = config.val_batch

    if dataset == RWF_DATASET:
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

    # prepare
    criterion = nn.CrossEntropyLoss().to(device)
    
    val_loss, val_acc = val(1, val_loader, model, criterion, device, val_log=None)
    print("Validation loss: {}, Accuray: {}".format(val_loss, val_acc))
    

def train_one_stream(config, home_path):
    # load model
    model, params = get_model(config, home_path)

    # dataset
    dataset = config.dataset
    stride = config.stride
    sample_duration = config.sample_duration

    # cross validation phase
    cv = config.num_cv
    input_mode = config.input_type

    sample_size, norm = build_transforms_parameters(model_type=config.model)
    
    # train set
    crop_method = GroupRandomScaleCenterCrop(size=sample_size)
    train_temporal_transform = build_temporal_transformation(config, config.train_temporal_transform)
    spatial_transform = Compose([crop_method,
                                 GroupRandomHorizontalFlip(),
                                 ToTensor(), 
                                 norm])
    target_transform = Label()

    train_batch = config.train_batch
    if dataset == RWF_DATASET:
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

    # prepare
    criterion = nn.CrossEntropyLoss().to(device)
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
        train_loss, train_acc, lr = train(i, train_loader, model, criterion, optimizer, device, batch_log, epoch_log)    
        epoch = i+1
        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('training accuracy', train_acc, epoch)

        scheduler.step(train_loss)
        if train_loss < loss_baseline:
            torch.save(model.state_dict(),
                       os.path.join(home_path, PATH_CHECKPOINT,'{}_fps{}_{}_({})_fold({})_{}_{:.4f}_{:.6f}.pth'.format(config.model,
                                                                                                            sample_duration,
                                                                                                            dataset,
                                                                                                            config.input_type.replace(' ','-'),
                                                                                                            cv,
                                                                                                            epoch,
                                                                                                            train_acc,
                                                                                                            train_loss)
                                    )
                        )
            loss_baseline = train_loss



if __name__ == '__main__':
    config = rwf_config()
    # config = hockey_config(config)
    
    # config.temp_annotation_path = os.path.join(environment_config['home'], PATH_SCORES,
    #     "Scores-dataset(rwf-2000)-ANmodel(AnomalyDetector_Dataset(UCFCrime2LocalClips)_Features(c3d)_TotalEpochs(100000)_ExtraInfo(c3d)-Epoch-10000)-input(rgb)")

    ##### For 2D CNN ####
    # config.num_epoch = 50
    # config.sample_size = (224,224)
    # config.sample_duration = 16
    # config.stride = 1 #for dynamic images it's frames to skip into a segment
    # config.ft_begin_idx = 0 # 0: train all layers, -1: freeze conv layers
    # config.acc_baseline = 0.90
    # config.additional_info = "-DI-STREAM"

    if config.dataset == RWF_DATASET:
        config.num_cv = 1
        # train_stream(config, environment_config['home'])

        # pretrained_model_rgb = '/media/david/datos/Violence DATA/VioNet_pth/i3d_fps16_rwf-20001_15_0.8356_0.394731.pth'
        # pretrained_model_di = '/media/david/datos/Violence DATA/VioNet_pth/i3d_fps16_rwf_2000_(dynamic-images)_1_15_0.8044_0.426578.pth'
        # val_one_stream(config, environment_config['home'])
        # val_two_stream(config, environment_config['home'], pretrained_model_rgb, pretrained_model_di)
        train_two_stream(config, environment_config['home'])
    elif config.dataset == HOCKEY_DATASET:
        
        for i in range(1,2):
            config.num_cv = i
            # pretrained_model_rgb = '/media/david/datos/Violence DATA/VioNet_pth/i3d_fps16_hockey_(rgb)_fold(5)_20_0.9938_0.041141.pth'
            # pretrained_model_di = '/media/david/datos/Violence DATA/VioNet_pth/i3d_fps16_hockey_(dynamic-images)_fold(5)_19_0.9925_0.039272.pth'
            
            # config.pretrained_model = pretrained_model_rgb
            # val_one_stream(config, environment_config['home'])
            
            # train_stream(config, environment_config['home'])

            # val_two_stream(config, environment_config['home'], pretrained_model_rgb, pretrained_model_di)
            train_two_stream(config, environment_config['home'])
