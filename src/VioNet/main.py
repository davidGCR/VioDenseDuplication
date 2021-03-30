import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from epoch import train, val, test
from model import VioNet_C3D, VioNet_ConvLSTM, VioNet_densenet, VioNet_densenet_lean, VioNet_Resnet, VioNet_Densenet2D
from dataset import VioDB
from config import Config

from transformations.spatial_transforms import Compose, ToTensor, Normalize
from transformations.spatial_transforms import GroupRandomHorizontalFlip, GroupRandomScaleCenterCrop, GroupScaleCenterCrop
from transformations.temporal_transforms import CenterCrop, RandomCrop, SegmentsCrop, RandomSegmentsCrop, KeyFrameCrop, TrainGuidedKeyFrameCrop, ValGuidedKeyFrameCrop, KeySegmentCrop
from transformations.target_transforms import Label, Video

from utils import Log
from torch.utils.tensorboard import SummaryWriter
from global_var import getFolder

g_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('main g_path:', g_path)



def main(config):
    # load model
    if config.model == 'c3d':
        model, params = VioNet_C3D(config)
    elif config.model == 'convlstm':
        model, params = VioNet_ConvLSTM(config)
    elif config.model == 'densenet':
        model, params = VioNet_densenet(config)
    elif config.model == 'densenet_lean':
        model, params = VioNet_densenet_lean(config)
    elif config.model == 'resnet50':
        model, params = VioNet_Resnet(config)
    elif config.model == 'densenet2D':
        model, params = VioNet_Densenet2D(config)
    # default densenet
    else:
        model, params = VioNet_densenet_lean(config)

    # dataset
    dataset = config.dataset
    sample_size = config.sample_size
    stride = config.stride
    sample_duration = config.sample_duration

    # cross validation phase
    cv = config.num_cv
    input_mode = config.input_mode
    train_temp_transform = config.train_temporal_transform
    val_temp_transform = config.val_temporal_transform
    # train set
    crop_method = GroupRandomScaleCenterCrop(size=sample_size)
    
    
    if input_mode == 'rgb':
        norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    elif input_mode == 'dynamic-images':
        norm = Normalize([0.49778724, 0.49780366, 0.49776983], [0.09050678, 0.09017131, 0.0898702 ])
        

    if train_temp_transform == 'random-crop':
        temporal_transform = RandomCrop(size=sample_duration, stride=stride, input_type=input_mode)
    elif train_temp_transform == 'segments':
        temporal_transform = SegmentsCrop(size=sample_duration, segment_size=config.segment_size, stride=stride, overlap=0.5)
    elif train_temp_transform == 'center-crop':
        temporal_transform = CenterCrop(size=sample_duration, stride=stride, input_type=input_mode)
    #     temporal_transform = TrainKeyFrameCrop(size=config.segment_size, stride=stride)
    # elif temp_transform == 'random-segments':
    #     temporal_transform = RandomSegmentsCrop(size=sample_duration, segment_size=15, stride=stride, overlap=0.5)
    elif train_temp_transform == 'keyframe':
        temporal_transform = KeyFrameCrop(size=sample_duration, stride=stride, input_type=input_mode)
    elif train_temp_transform == 'guided-segment':
        temporal_transform = TrainGuidedKeyFrameCrop(size=sample_duration, segment_size=config.segment_size, stride=stride, overlap=0.5)
    elif train_temp_transform == 'keysegment':
        temporal_transform = KeySegmentCrop(size=sample_duration, stride=stride, input_type=input_mode, segment_type="highestscore")




    spatial_transform = Compose(
        [crop_method,
         GroupRandomHorizontalFlip(),
         ToTensor(), norm])
    target_transform = Label()

    train_batch = config.train_batch
    if dataset == 'rwf-2000':
      train_data = VioDB(g_path + '/VioDB/{}_jpg/frames/'.format(dataset),
                       g_path + '/VioDB/{}_jpg{}.json'.format(dataset, cv), 'training',
                       spatial_transform, temporal_transform, target_transform, dataset,
                       tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path))
    else:
      train_data = VioDB(g_path + '/VioDB/{}_jpg/'.format(dataset),
                        g_path + '/VioDB/{}_jpg{}.json'.format(dataset, cv), 'training',
                        spatial_transform, temporal_transform, target_transform)
    train_loader = DataLoader(train_data,
                              batch_size=train_batch,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

    # val set
    crop_method = GroupScaleCenterCrop(size=sample_size)
    
    
    if input_mode == 'rgb':
        norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    elif input_mode == 'dynamic-images':
        norm = Normalize([0.49778724, 0.49780366, 0.49776983], [0.09050678, 0.09017131, 0.0898702 ])
        
    
    if val_temp_transform == 'standar':
        temporal_transform = CenterCrop(size=sample_duration, stride=stride)
    elif val_temp_transform == 'segments':
        temporal_transform = SegmentsCrop(size=sample_duration, segment_size=config.segment_size, stride=stride, overlap=0.5)
    elif val_temp_transform == 'center-crop':
        temporal_transform = CenterCrop(size=sample_duration, stride=stride, input_type=input_mode)
    # elif temp_transform == 'random-segments':
    #     temporal_transform = SegmentsCrop(size=sample_duration, segment_size=15, stride=stride, overlap=0.5)
    elif val_temp_transform == 'keyframe':
        temporal_transform = KeyFrameCrop(size=sample_duration, stride=stride, input_type=input_mode)
    elif val_temp_transform == 'guided-segment':
        temporal_transform = ValGuidedKeyFrameCrop(size=sample_duration, segment_size=config.segment_size, stride=stride, overlap=0.5)
    elif val_temp_transform == 'keysegment':
        temporal_transform = KeySegmentCrop(size=sample_duration, stride=stride, input_type=input_mode, segment_type="highestscore")

    spatial_transform = Compose([crop_method, ToTensor(), norm])
    target_transform = Label()

    val_batch = config.val_batch

    if dataset == 'rwf-2000':
      val_data = VioDB(g_path + '/VioDB/{}_jpg/frames/'.format(dataset),
                     g_path + '/VioDB/{}_jpg{}.json'.format(dataset, cv), 'validation',
                     spatial_transform, temporal_transform, target_transform, dataset,
                     tmp_annotation_path=os.path.join(g_path, config.temp_annotation_path))
    else:
      val_data = VioDB(g_path + '/VioDB/{}_jpg/'.format(dataset),
                      g_path + '/VioDB/{}_jpg{}.json'.format(dataset, cv), 'validation',
                      spatial_transform, temporal_transform, target_transform)
    val_loader = DataLoader(val_data,
                            batch_size=val_batch,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    log_path = getFolder('VioNet_log')
    chk_path = getFolder('VioNet_pth')
    tsb_path = getFolder('VioNet_tensorboard_log')

    log_tsb_dir = tsb_path + '/{}_fps{}_{}_split{}_input({})_Info({})'.format(config.model, sample_duration,
                                                dataset, cv, input_mode, config.additional_info)
    for pth in [log_path, chk_path, tsb_path, log_tsb_dir]:
        # make dir
        if not os.path.exists(pth):
            os.mkdir(pth)

    print('tensorboard dir:', log_tsb_dir)                                                
    writer = SummaryWriter(log_tsb_dir)

    # log
    batch_log = Log(
        log_path+'/{}_fps{}_{}_batch{}_input({})_Info({}).log.csv'.format(
            config.model,
            sample_duration,
            dataset,
            cv,
            input_mode, config.additional_info
        ), ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    epoch_log = Log(
        log_path+'/{}_fps{}_{}_epoch{}_input({})_Info({}).log.csv'.format(config.model, sample_duration,
                                               dataset, cv, input_mode, config.additional_info),
        ['epoch', 'loss', 'acc', 'lr'])
    val_log = Log(
        log_path+'/{}_fps{}_{}_val{}_input({})_Info({}).log.csv'.format(config.model, sample_duration,
                                             dataset, cv, input_mode, config.additional_info),
        ['epoch', 'loss', 'acc'])
    
    train_val_log = Log(log_path+'/{}_fps{}_{}_split{}_input({})_Info({}).LOG.csv'.format(config.model, sample_duration,
                                               dataset, cv, input_mode, config.additional_info),
        ['epoch', 'train_loss', 'train_acc', 'lr', 'val_loss', 'val_acc'])

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

    # for i, (inputs, targets) in enumerate(val_loader):
    #     print('inputs:', inputs.size())

    for i in range(config.num_epoch):
        train_loss, train_acc, lr = train(i, train_loader, model, criterion, optimizer, device, batch_log,
              epoch_log)
        val_loss, val_acc = val(i, val_loader, model, criterion, device,
                                val_log)
        epoch = i+1
        train_val_log.log({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'lr': lr, 'val_loss': val_loss, 'val_acc': val_acc})
        writer.add_scalar('training loss',
                            train_loss,
                            epoch)
        writer.add_scalar('training accuracy',
                            train_acc,
                            epoch)
        writer.add_scalar('validation loss',
                            val_loss,
                            epoch)
        writer.add_scalar('validation accuracy',
                            val_acc,
                            epoch)

        scheduler.step(val_loss)
        if val_acc > acc_baseline or (val_acc >= acc_baseline and
                                      val_loss < loss_baseline):
            torch.save(
                model.state_dict(),
                chk_path+'/{}_fps{}_{}{}_{}_{:.4f}_{:.6f}.pth'.format(
                    config.model, sample_duration, dataset, cv, i, val_acc,
                    val_loss))
            acc_baseline = val_acc
            loss_baseline = val_loss


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = 'hockey'
    config = Config(
        'densenet2D',  # c3d, convlstm, densenet, densenet_lean, resnet50, densenet2D
        dataset,
        device=device,
        num_epoch=50,
        acc_baseline=0.92,
        ft_begin_idx=0,
    )

    # train params for different datasets
    configs = {
        'hockey': {
            'lr': 1e-2,
            'batch_size': 32
        },
        'movie': {
            'lr': 1e-3,
            'batch_size': 16
        },
        'vif': {
            'lr': 1e-3,
            'batch_size': 16
        },
        'rwf-2000': {
            'lr': 1e-3,
            'batch_size': 32
        }
    }

    # for dataset in ['rwf-2000','hockey', 'movie', 'vif']:
    # config.dataset = dataset
    config.train_batch = configs[dataset]['batch_size']
    config.val_batch = configs[dataset]['batch_size']
    config.learning_rate = configs[dataset]['lr']
    config.input_mode = 'dynamic-images' #rgb, dynamic-images
    config.train_temporal_transform = 'keysegment' #standar, segments, segments-keyframe, random-segments, keyframe, guided-segment, keysegment
    config.val_temporal_transform = 'keysegment'
    config.temp_annotation_path = ""
    # 5 fold cross validation
    # for cv in range(1, 6):
    #     config.num_cv = cv
    #     main(config)

    ##### For 2D CNN ####
    # config.num_epoch = 50
    config.sample_size = (224,224)
    config.sample_duration = 5
    config.stride = 1 #for dynamic images it's frames to skip into a segment
    config.ft_begin_idx = 0 # 0: train all layers, -1: freeze conv layers
    config.acc_baseline = 0.90
    config.additional_info = "central-segment-changed_tmp-transf-5frames"

    config.num_cv = 1
    main(config)