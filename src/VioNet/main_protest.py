import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from epoch import train, val, test
from model import VioNet_C3D, VioNet_ConvLSTM, VioNet_densenet, VioNet_densenet_lean, VioNet_Resnet, VioNet_Densenet2D
from dataset import VioDB, ProtestDataset
from config import Config

from spatial_transforms import Compose, ToTensor, Normalize
from spatial_transforms import GroupRandomHorizontalFlip, GroupRandomScaleCenterCrop, GroupScaleCenterCrop, Lighting
from temporal_transforms import OneFrameCrop
from target_transforms import Label, Video
import torchvision.transforms as transforms

from utils import Log
from torch.utils.tensorboard import SummaryWriter
from global_var import getFolder

g_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print('main g_path:', g_path)

def load_protest_dataset(config):

    # data_dir = "/Users/davidchoqueluqueroman/Documents/CODIGOS/DATASETS/UCLA-protest"
    data_dir = g_path+"/VioDB/UCLA-protest"
    img_dir_train = os.path.join(data_dir, "img/train")
    img_dir_val = os.path.join(data_dir, "img/test")
    txt_file_train = os.path.join(data_dir, "annot_train.txt")
    txt_file_val = os.path.join(data_dir, "annot_test.txt")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    eigvec = torch.Tensor([[-0.5675,  0.7192,  0.4009],
                            [-0.5808, -0.0045, -0.8140],
                            [-0.5836, -0.6948,  0.4203]])


    transform = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomRotation(30),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(
                            brightness = 0.4,
                            contrast = 0.4,
                            saturation = 0.4,
                            ),
                        transforms.ToTensor(),
                        Lighting(0.1, eigval, eigvec),
                        normalize,
                    ])

    train_batch = config.train_batch

    train_dataset = ProtestDataset(txt_file_train, img_dir_train, transform)
    train_loader = DataLoader(
                    train_dataset,
                    num_workers = 4,
                    batch_size = train_batch,
                    shuffle = True
                    )

    #validation
    val_batch = config.val_batch
    val_dataset = ProtestDataset(
                    txt_file = txt_file_val,
                    img_dir = img_dir_val,
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]))
    
    val_loader = DataLoader(
                    val_dataset,
                    num_workers = 4,
                    batch_size = val_batch,
                    shuffle=False)
    
    return train_loader, val_loader

def load_hockey_dataset(config):
    dataset = config.dataset
    # stride = config.stride
    # sample_duration = config.sample_duration

    cv = config.num_cv

    train_batch = config.train_batch
    sample_size = config.sample_size

    spatial_transform = Compose(
        [crop_method,
         GroupRandomHorizontalFlip(),
         ToTensor(), norm])
    target_transform = Label()
    temporal_transform = OneFrameCrop(position=0)
    train_data = VioDB(g_path + '/VioDB/{}_jpg/'.format(dataset),
                        g_path + '/VioDB/{}_jpg{}.json'.format(dataset, cv), 'training',
                        spatial_transform, temporal_transform, target_transform)
    train_loader = DataLoader(train_data,
                              batch_size=train_batch,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    
    val_batch = config.val_batch
    crop_method = GroupScaleCenterCrop(size=sample_size)
    spatial_transform = Compose([crop_method, ToTensor(), norm])
    target_transform = Label()
    # temporal_transform = OneFrameCrop(position=1)

    val_data = VioDB(g_path + '/VioDB/{}_jpg/'.format(dataset),
                      g_path + '/VioDB/{}_jpg{}.json'.format(dataset, cv), 'validation',
                      spatial_transform, temporal_transform, target_transform)
    val_loader = DataLoader(val_data,
                            batch_size=val_batch,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    return train_loader, val_loader
    

def main(config):
    # load model
    if config.model == 'resnet50':
        model, params = VioNet_Resnet(config)
    elif config.model == 'densenet2D':
        model, params = VioNet_Densenet2D(config)
    # dataset
    dataset = config.dataset
    stride = config.stride
    sample_duration = config.sample_duration

    # cross validation phase
    cv = config.num_cv
    input_mode = config.input_mode
    temp_transform = config.temporal_transform
    
    if dataset == "protest":
        train_loader, val_loader = load_protest_dataset(config)
    elif dataset == "hockey":
        train_loader, val_loader = load_hockey_dataset(config)
  

    log_path = getFolder('VioNet_log')
    chk_path = getFolder('VioNet_pth')
    tsb_path = getFolder('VioNet_tensorboard_log')

    log_tsb_dir = tsb_path + '/{}_fps{}_{}_split{}_input({})_tempTransform({})_Info({})'.format(config.model, sample_duration,
                                                dataset, cv, input_mode, temp_transform, config.additional_info)
    for pth in [log_path, chk_path, tsb_path, log_tsb_dir]:
        # make dir
        if not os.path.exists(pth):
            os.mkdir(pth)

    print('tensorboard dir:', log_tsb_dir)                                                
    writer = SummaryWriter(log_tsb_dir)

    # log
    batch_log = Log(
        log_path+'/{}_fps{}_{}_batch{}_input({})_tempTransform({})_Info({}).log.csv'.format(
            config.model,
            sample_duration,
            dataset,
            cv,
            input_mode, temp_transform, config.additional_info
        ), ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    epoch_log = Log(
        log_path+'/{}_fps{}_{}_epoch{}_input({})_tempTransform({})_Info({}).log.csv'.format(config.model, sample_duration,
                                               dataset, cv, input_mode, temp_transform, config.additional_info),
        ['epoch', 'loss', 'acc', 'lr'])
    val_log = Log(
        log_path+'/{}_fps{}_{}_val{}_input({})_tempTransform({})_Info({}).log.csv'.format(config.model, sample_duration,
                                             dataset, cv, input_mode, temp_transform, config.additional_info),
        ['epoch', 'loss', 'acc'])
    
    train_val_log = Log(log_path+'/{}_fps{}_{}_split{}_input({})_tempTransform({})_Info({}).LOG.csv'.format(config.model, sample_duration,
                                               dataset, cv, input_mode, temp_transform, config.additional_info),
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
        'resnet50',  # c3d, convlstm, densenet, densenet_lean, resnet50, densenet2D
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
        },
        'protest': {
            'lr': 1e-3,
            'batch_size': 32
        }
    }

    # for dataset in ['rwf-2000','hockey', 'movie', 'vif']:
    # config.dataset = dataset
    config.train_batch = configs[dataset]['batch_size']
    config.val_batch = configs[dataset]['batch_size']
    config.learning_rate = configs[dataset]['lr']
    config.input_mode = 'rgb' #rgb, dynamic-images
    config.pretrained_model = "resnet50_fps1_protest1_38_0.9757_0.073047.pth"
   
    ##### For 2D CNN ####
    config.sample_size = (224,224)
    config.sample_duration =  1# Number of dynamic images
    config.stride = 1 #for dynamic images it's frames to skip into a segment
    config.ft_begin_idx = 0 # 0: train all layers, -1: freeze conv layers
    config.additional_info = "finetuned-with-hockey"

    config.num_cv = 1
    main(config)