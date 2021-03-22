import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from epoch import train, val, test

from model import VioNet_Resnet, VioNet_Densenet2D
from config import Config

from spatial_transforms import Compose, ToTensor, Normalize, DIPredefinedTransforms
from temporal_transforms import DynamicImage
from target_transforms import Label, Video
import torchvision.transforms as transforms
from video_dataset import HMDB51DatasetV2
import torchvision

from utils import Log
from torch.utils.tensorboard import SummaryWriter
from global_var import getFolder

g_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def laod_HMDB51_dataset(config: Config, root, annotation_path):
    DN = DynamicImage(output_type="pil")
    # mean = [0.49778724, 0.49780366, 0.49776983]
    # std = [0.09050678, 0.09017131, 0.0898702]
    mean=None
    std=None
    transformations = DIPredefinedTransforms(size=224, tmp_transform=DN, mean=mean, std=std)

    hmdb51_data_train = HMDB51DatasetV2(root=root,
                                        annotation_path=annotation_path,
                                        frames_per_clip=config.sample_duration,
                                        step_between_clips=config.stride,
                                        fold=config.num_cv,
                                        train=True,
                                        transform=transformations.train_transform)
    train_data_loader = torch.utils.data.DataLoader(hmdb51_data_train,
                                            batch_size=config.train_batch,
                                            shuffle=True,
                                            num_workers=4)

    hmdb51_data_val = HMDB51DatasetV2(root=root,
                                    annotation_path=annotation_path,
                                    frames_per_clip=config.sample_duration,
                                    step_between_clips=config.stride,
                                    fold=config.num_cv,
                                    train=False,
                                    transform=transformations.val_transform)                                              
    val_data_loader = torch.utils.data.DataLoader(hmdb51_data_val,
                                            batch_size=config.val_batch,
                                            shuffle=False,
                                            num_workers=4)
    print("Train set:", len(hmdb51_data_train))
    print("Val set:", len(hmdb51_data_val))                                        
    return train_data_loader, val_data_loader

def main(config: Config, root, annotation_path):
    if config.model == 'resnet50':
        model, params = VioNet_Resnet(config)
    elif config.model == 'densenet2D':
        model, params = VioNet_Densenet2D(config)
    log_path = getFolder('VioNet_log')
    chk_path = getFolder('VioNet_pth')
    tsb_path = getFolder('VioNet_tensorboard_log')

    log_tsb_dir = tsb_path + '/{}_fps{}_{}_split{}_input({})_Info({})'.format(config.model,
                                                                              config.sample_duration,
                                                                              config.dataset,
                                                                              config.num_cv,
                                                                              config.input_mode,
                                                                              config.additional_info)
    for pth in [log_path, chk_path, tsb_path, log_tsb_dir]:
        # make dir
        if not os.path.exists(pth):
            os.mkdir(pth)

    print('tensorboard dir:', log_tsb_dir)                                                
    writer = SummaryWriter(log_tsb_dir)

    # log
    batch_log = Log(
        log_path+'/{}_fps{}_{}_batch{}_input({})_Info({}).log.csv'.format(config.model,
                                                                          config.sample_duration,
                                                                          config.dataset,
                                                                          config.num_cv,
                                                                          config.input_mode,
                                                                          config.additional_info),
        ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    epoch_log = Log(
        log_path+'/{}_fps{}_{}_epoch{}_input({})_Info({}).log.csv'.format(config.model, 
                                                                          config.sample_duration,
                                                                          config.dataset,
                                                                          config.num_cv,
                                                                          config.input_mode,
                                                                          config.additional_info),
        ['epoch', 'loss', 'acc', 'lr'])
    val_log = Log(
        log_path+'/{}_fps{}_{}_val{}_input({})_Info({}).log.csv'.format(config.model,
                                                                        config.sample_duration,
                                                                        config.dataset,
                                                                        config.num_cv,
                                                                        config.input_mode,
                                                                        config.additional_info),
        ['epoch', 'loss', 'acc'])
    
    train_val_log = Log(log_path+'/{}_fps{}_{}_split{}_input({})_Info({}).LOG.csv'.format(config.model,
                                                                                          config.sample_duration,
                                                                                          config.dataset,
                                                                                          config.num_cv,
                                                                                          config.input_mode,
                                                                                          config.additional_info),
        ['epoch', 'train_loss', 'train_acc', 'lr', 'val_loss', 'val_acc'])


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

    train_loader, val_loader = laod_HMDB51_dataset(config, root, annotation_path)

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
                    config.model, config.sample_duration, config.dataset, config.num_cv, i, val_acc,
                    val_loss))
            acc_baseline = val_acc
            loss_baseline = val_loss


if __name__ == "__main__":
    
    dataset = 'hmdb51'
    config = Config(
        'resnet50',  # c3d, convlstm, densenet, densenet_lean, resnet50, densenet2D
        dataset,
        device=device,
        num_epoch=50,
        acc_baseline=0.70,
        ft_begin_idx=0,
    )

    # train params for different datasets
    configs = {
        'hockey': {
            'lr': 1e-2,
            'batch_size': 32
        },
        'hmdb51': {
            'lr': 3e-3,
            'batch_size': 32
        }
    }

    # for dataset in ['rwf-2000','hockey', 'movie', 'vif']:
    # config.dataset = dataset
    config.train_batch = configs[dataset]['batch_size']
    config.val_batch = configs[dataset]['batch_size']
    config.learning_rate = configs[dataset]['lr']
    config.input_mode = 'dynamic-images' #rgb, dynamic-images
    # config.pretrained_model = "resnet50_fps1_protest1_38_0.9757_0.073047.pth"
   
    ##### For 2D CNN ####
    config.num_classes = 51
    config.sample_size = (224,224)
    config.sample_duration =  10# Number of frames to compute Dynamic images
    config.stride = 100 #It means number of frames to skip in a video between video clips
    config.ft_begin_idx = 0 # 0: train all layers, -1: freeze conv layers
    config.additional_info = ""
    
    # root='/Users/davidchoqueluqueroman/Documents/CODIGOS/DATASETS_Local/hmdb51/hmdb51_org'
    # annotation_path='/Users/davidchoqueluqueroman/Documents/CODIGOS/DATASETS_Local/hmdb51/testTrainMulti_7030_splits'
    root='/content/DATASETS/HMDB51'
    annotation_path='/content/drive/MyDrive/VIOLENCE DATA/HMDB51/testTrainMulti_7030_splits'
    print(os.listdir(annotation_path))
    config.num_cv = 1
    main(config, root, annotation_path)
