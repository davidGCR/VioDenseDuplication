
from os import path
from pickle import FALSE
from PIL import Image, ImageFile

# from VioNet.dataset import make_dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch._C import device
from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision.transforms.transforms import Resize
import torchvision.transforms as transforms
import os

#data
from customdatasets.make_dataset import MakeRWF2000, MakeHockeyDataset
from customdatasets.tube_dataset import TubeDataset, my_collate, my_collate_2, OneVideoTubeDataset, TubeFeaturesDataset
from torch.utils.data import DataLoader, dataset

from config import Config
from model import ViolenceDetector_model, VioNet_I3D_Roi, VioNet_densenet_lean_roi
from models.anomaly_detector import custom_objective, RegularizedLoss
from epoch import calculate_accuracy_2, train_regressor
from utils import Log, save_checkpoint, load_checkpoint
from utils import get_torch_device
from feature_writer2 import FeaturesWriter
from models.anomaly_detector import custom_objective
from torch import nn
from epoch import train, val
import numpy as np
from global_var import *
from configs_datasets import DefaultTrasformations
from models.mil_loss import MIL

def load_features(config: Config):
    device = config.device
    make_dataset = MakeRWF2000(root='/media/david/datos/Violence DATA/RWF-2000/frames',
                                train=False,
                                path_annotations='/media/david/datos/Violence DATA/Tubes/RWF-2000',
                                path_feat_annotations='/media/david/datos/Violence DATA/i3d-FeatureMaps/rwf'
                                )
    dataset = TubeFeaturesDataset(frames_per_tube=16,
                                    min_frames_per_tube=8,
                                    make_function=make_dataset,
                                    map_shape=(1,528,4,14,14),
                                    max_num_tubes=4)
    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=4,
                        # pin_memory=True,
                        )
    for i, data in enumerate(loader):
        boxes, f_maps, label = data
        print('boxes: ', boxes.size())
        print('f_maps: ', f_maps.size())
        print('label: ', label.size())

def extract_features(confi: Config, output_folder: str):
    device = config.device
    make_dataset = MakeRWF2000(root='/media/david/datos/Violence DATA/RWF-2000/frames',
                                train=False,
                                path_annotations='/media/david/datos/Violence DATA/Tubes/RWF-2000')
    paths, labels, annotations, _ = make_dataset()
    from models.i3d import InceptionI3d
    model = InceptionI3d(2, in_channels=3, final_endpoint='Mixed_4e').to(device)
    load_model_path = '/media/david/datos/Violence DATA/VioNet_weights/pytorch_i3d/rgb_imagenet.pt'
    state_dict = torch.load(load_model_path)
    model.load_state_dict(state_dict,  strict=False)
    model.eval()

    from models.roi_extractor_3d import SingleRoIExtractor3D
    roi_op = SingleRoIExtractor3D(roi_layer_type='RoIAlign',
                                featmap_stride=16,
                                output_size=8,    
                                with_temporal_pool=True).to(device)
    
    features_writer = FeaturesWriter(num_videos=len(paths), num_segments=0)

    with torch.no_grad():
        for i in range(len(paths)):
            video_path = paths[i]
            annotation_path = annotations[i]
            print("==={}/{}".format(i+1, video_path))
            dataset = OneVideoTubeDataset(frames_per_tube=16, 
                                            min_frames_per_tube=8, 
                                            video_path=video_path,
                                            annotation_path=annotation_path,
                                            spatial_transform=transforms.Compose([
                                                # transforms.CenterCrop(224),
                                                # transforms.Resize(256),
                                                transforms.ToTensor()
                                            ]),
                                            max_num_tubes=0)
            loader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4,
                                # pin_memory=True,
                                # collate_fn=my_collate
                                )
            print('loader len:', len(loader))
            if len(loader)==0:
                continue
            tmp_names = video_path.split('/')
            output_file = os.path.join(output_folder, tmp_names[-3], tmp_names[-2], tmp_names[-1])# split, class, video

            for j, data in enumerate(loader):
                box, tube_images = data
                tube_images = tube_images.permute(0,2,1,3,4)
                box = torch.squeeze(box, dim=1)
                
                if box ==None:
                    print("Noneeee")
                    continue
                
                box = box.to(device)
                tube_images = tube_images.to(device)
                # print('box: ', box.size())
                print('tube_images: ', tube_images.size(), tube_images.device)

                f_map = model(tube_images)
                print('f_map: ', f_map.size())
                features_writer.write(feature=torch.flatten(f_map).cpu().numpy(),
                                        video_name=tmp_names[-1],
                                        idx=j,
                                        dir=os.path.join(output_folder, tmp_names[-3], tmp_names[-2]))
            features_writer.dump()

def load_make_dataset(dataset_name, train=True, cv_split=1, home_path='', category=2):
    if dataset_name == RWF_DATASET:
        make_dataset = MakeRWF2000(root=os.path.join(home_path, 'RWF-2000/frames'),#'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames', 
                                    train=train,
                                    category=category, 
                                    path_annotations=os.path.join(home_path, 'ActionTubes/RWF-2000'))#'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/Tubes/RWF-2000')

    elif dataset_name == HOCKEY_DATASET:
        make_dataset = MakeHockeyDataset(root=os.path.join(home_path, 'HockeyFightsDATASET/frames'), #'/content/DATASETS/HockeyFightsDATASET/frames'
                                        train=train,
                                        cv_split_annotation_path=os.path.join(home_path, 'VioNetDB-splits/hockey_jpg{}.json'.format(cv_split)), #'/content/DATASETS/VioNetDB-splits/hockey_jpg{}.json'
                                        path_annotations=os.path.join(home_path, 'ActionTubes/hockey'))#'/content/DATASETS/ActionTubes/hockey'
    return make_dataset


def main(config: Config):
    device = config.device
    make_dataset = load_make_dataset(config.dataset, train=True, cv_split=config.num_cv, home_path=config.home_path, category=2)
    spatial_t = DefaultTrasformations(model_name=config.model, size=224, train=True)
    dataset = TubeDataset(frames_per_tube=16, 
                            min_frames_per_tube=8,
                            make_function=make_dataset,
                            spatial_transform=spatial_t(),
                            max_num_tubes=config.num_tubes,
                            train=True,
                            dataset=config.dataset,
                            input_type=config.input_type,
                            random=config.tube_sampling_random)
    loader = DataLoader(dataset,
                        batch_size=config.train_batch,
                        shuffle=True,
                        num_workers=4,
                        # pin_memory=True,
                        collate_fn=my_collate
                        )

    #validation
    val_make_dataset = load_make_dataset(config.dataset, train=False, cv_split=config.num_cv, home_path=config.home_path)
    spatial_t_val = DefaultTrasformations(model_name='densenet_lean_roi', size=224, train=False)
    val_dataset = TubeDataset(frames_per_tube=16, 
                            min_frames_per_tube=8, 
                            make_function=val_make_dataset,
                            spatial_transform=spatial_t_val(),
                            max_num_tubes=config.num_tubes,
                            train=False,
                            dataset=config.dataset,
                            input_type=config.input_type,
                            random=config.tube_sampling_random)
    val_loader = DataLoader(val_dataset,
                        batch_size=config.val_batch,
                        shuffle=True,
                        num_workers=1,
                        # pin_memory=True,
                        collate_fn=my_collate
                        )
   
    ################## Full Detector ########################
    
    #
    from models.violence_detector import ViolenceDetectorBinary
    if config.model == 'densenet_lean_roi':
        model, params = VioNet_densenet_lean_roi(config, config.pretrained_model)
    elif config.model == 'i3d+roi+fc':
        model, params = ViolenceDetector_model(config, device, config.pretrained_model)
    elif config.model == 'i3d+roi+i3d':
        model, params = VioNet_I3D_Roi(config, device, config.pretrained_model)
    elif config.model == 'i3d+roi+binary':
        model = ViolenceDetectorBinary(
            freeze=config.freeze,
            input_dim=528).to(device)
        params = model.parameters()

    exp_config_log = "SpTmpDetector_{}_model({})_head({})_stream({})_cv({})_epochs({})_tubes({})_tub_sampl_rand({})_optimizer({})_lr({})_note({})".format(config.dataset,
                                                                config.model,
                                                                config.head,
                                                                config.input_type,
                                                                config.num_cv,
                                                                config.num_epoch,
                                                                config.num_tubes,
                                                                config.tube_sampling_random,
                                                                config.optimizer,
                                                                config.learning_rate,
                                                                config.additional_info)
    
    h_p = HOME_DRIVE if config.home_path==HOME_COLAB else config.home_path
    tsb_path_folder = os.path.join(h_p, PATH_TENSORBOARD, exp_config_log)
    chk_path_folder = os.path.join(h_p, PATH_CHECKPOINT, exp_config_log)

    for p in [tsb_path_folder, chk_path_folder]:
        if not os.path.exists(p):
            os.makedirs(p)
    # print('tensorboard dir:', tsb_path)                                                
    writer = SummaryWriter(tsb_path_folder)

    if config.optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(params, lr=config.learning_rate, eps=1e-8)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=params,
                                    lr=config.learning_rate,
                                    momentum=0.5,
                                    weight_decay=1e-3)
    
    if config.head == REGRESSION:
        # criterion = nn.BCELoss().to(device)
        # criterion = nn.BCEWithLogitsLoss().to(device)
        criterion = MIL
    elif config.head == BINARY:
        criterion = nn.CrossEntropyLoss().to(config.device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           verbose=True,
                                                           factor=config.factor,
                                                           min_lr=config.min_lr)
    from utils import AverageMeter
    # from epoch import calculate_accuracy_2

    start_epoch = 0
    ##Restore training
    if config.restore_training:
        model, optimizer, epochs, last_epoch, last_loss = load_checkpoint(model, config.device, optimizer, config.checkpoint_path)
        start_epoch = last_epoch+1
        # config.num_epoch = epochs
    
    def train(_epoch, _model, _criterion, _optimizer):
        print('training at epoch: {}'.format(epoch))
        _model.train()
        losses = AverageMeter()
        accuracies = AverageMeter()
        for i, data in enumerate(loader):
            boxes, video_images, labels, num_tubes, paths = data
            boxes, video_images = boxes.to(device), video_images.to(device)
            # labels = labels.float().to(device)
            labels = labels.float().to(device) if config.head == REGRESSION else labels.to(device)

            # print('video_images: ', video_images.size())
            # print('num_tubes: ', config.num_tubes)
            # print('boxes: ', boxes, boxes.size())

            # zero the parameter gradients
            optimizer.zero_grad()
            #predict
            outs = _model(video_images, boxes, config.num_tubes)
            #loss
            # print('outs: ', outs.size())
            # print('before criterion labels: ', labels, labels.size())
            
            loss = _criterion(outs,labels)
            #accuracy
            # preds = np.round(scores.detach().cpu().numpy())
            # acc = (preds == labels.cpu().numpy()).sum() / preds.shape[0]
            if config.head == REGRESSION:
                acc = get_accuracy(outs, labels)
            else:
                acc = calculate_accuracy_2(outs,labels)
            # meter
            # print('len(video_images): ', len(video_images), ' video_images.size(0):',video_images.size(0), ' preds.shape[0]:', preds.shape[0])
            losses.update(loss.item(), outs.shape[0])
            accuracies.update(acc, outs.shape[0])
            # backward + optimize
            loss.backward()
            _optimizer.step()
            # print(
            #     'Epoch: [{0}][{1}/{2}]\t'
            #     # 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #     # 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #     'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
            #         epoch,
            #         i + 1,
            #         len(loader),
            #         # batch_time=batch_time,
            #         # data_time=data_time,
            #         loss=losses,
            #         acc=accuracies
            #     )
            # )
        train_loss = losses.avg
        train_acc = accuracies.avg
        print(
            'Epoch: [{}]\t'
            'Loss(train): {loss.avg:.4f}\t'
            'Acc(train): {acc.avg:.3f}'.format(_epoch, loss=losses, acc=accuracies)
        )
        return train_loss, train_acc


    def val(_epoch, _model, _criterion):
        print('validation at epoch: {}'.format(epoch))
        # set model to evaluate mode
        _model.eval()
        # meters
        losses = AverageMeter()
        accuracies = AverageMeter()
        for _, data in enumerate(val_loader):
            boxes, video_images, labels, num_tubes, paths = data
            boxes, video_images = boxes.to(device), video_images.to(device)
            # labels = labels.float().to(device)
            labels = labels.float().to(device) if config.head == REGRESSION else labels.to(device)
            # no need to track grad in eval mode
            with torch.no_grad():
                outputs = _model(video_images, boxes, config.num_tubes)
                loss = _criterion(outputs, labels)
                # preds = np.round(outputs.detach().cpu().numpy())
                # acc = (preds == labels.cpu().numpy()).sum() / preds.shape[0]
                
                if config.head == REGRESSION:
                    acc = get_accuracy(outputs, labels)
                else:
                    acc = calculate_accuracy_2(outputs,labels)

            losses.update(loss.item(), outputs.shape[0])
            accuracies.update(acc, outputs.shape[0])

        print(
            'Epoch: [{}]\t'
            'Loss(val): {loss.avg:.4f}\t'
            'Acc(val): {acc.avg:.3f}'.format(_epoch, loss=losses, acc=accuracies)
        )
        val_loss = losses.avg
        val_acc = accuracies.avg

        return val_loss, val_acc

    for epoch in range(start_epoch, config.num_epoch):
        # epoch = last_epoch+i
        train_loss, train_acc = train(epoch, model, criterion, optimizer)
        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('training accuracy', train_acc, epoch)
        
        val_loss, val_acc = val(epoch, model, criterion)
        scheduler.step(val_loss)
        writer.add_scalar('validation loss', val_loss, epoch)
        writer.add_scalar('validation accuracy', val_acc, epoch)

        if (epoch+1)%config.save_every == 0:
            save_checkpoint(model, config.num_epoch, epoch, optimizer,train_loss, os.path.join(chk_path_folder,"save_at_epoch-"+str(epoch)+".chk"))

def main_2(config: Config):
    device = config.device
    make_dataset_nonviolence = load_make_dataset(
        config.dataset, 
        train=True,
        cv_split=config.num_cv, 
        home_path=config.home_path,
        category=0
        )
    make_dataset_violence = load_make_dataset(
        config.dataset, 
        train=True,
        cv_split=config.num_cv, 
        home_path=config.home_path,
        category=1
        )
    spatial_t = DefaultTrasformations(model_name=config.model, size=224, train=True)
    dataset_train_nonviolence = TubeDataset(frames_per_tube=16, 
                            min_frames_per_tube=8,
                            make_function=make_dataset_nonviolence,
                            spatial_transform=spatial_t(),
                            max_num_tubes=config.num_tubes,
                            train=True,
                            dataset=config.dataset,
                            input_type=config.input_type,
                            random=config.tube_sampling_random)
    dataset_train_violence = TubeDataset(frames_per_tube=16, 
                            min_frames_per_tube=8,
                            make_function=make_dataset_violence,
                            spatial_transform=spatial_t(),
                            max_num_tubes=config.num_tubes,
                            train=True,
                            dataset=config.dataset,
                            input_type=config.input_type,
                            random=config.tube_sampling_random)
    loader_train_nonviolence = DataLoader(dataset_train_nonviolence,
                        batch_size=config.train_batch,
                        shuffle=True,
                        num_workers=4,
                        # pin_memory=True,
                        collate_fn=my_collate
                        )
    loader_train_violence = DataLoader(dataset_train_violence,
                        batch_size=config.train_batch,
                        shuffle=True,
                        num_workers=4,
                        # pin_memory=True,
                        collate_fn=my_collate
                        )

    #validation
    val_make_dataset_nonviolence = load_make_dataset(
        config.dataset, 
        train=False,
        cv_split=config.num_cv, 
        home_path=config.home_path,
        category=0
        )
    val_make_dataset_violence = load_make_dataset(
        config.dataset, 
        train=False,
        cv_split=config.num_cv, 
        home_path=config.home_path,
        category=1
        )
    dataset_val_nonviolence = TubeDataset(frames_per_tube=16, 
                            min_frames_per_tube=8,
                            make_function=val_make_dataset_nonviolence,
                            spatial_transform=spatial_t(),
                            max_num_tubes=config.num_tubes,
                            train=False,
                            dataset=config.dataset,
                            input_type=config.input_type,
                            random=config.tube_sampling_random)
    dataset_val_violence = TubeDataset(frames_per_tube=16, 
                            min_frames_per_tube=8,
                            make_function=val_make_dataset_violence,
                            spatial_transform=spatial_t(),
                            max_num_tubes=config.num_tubes,
                            train=False,
                            dataset=config.dataset,
                            input_type=config.input_type,
                            random=config.tube_sampling_random)
    loader_val_nonviolence = DataLoader(dataset_val_nonviolence,
                        batch_size=config.train_batch,
                        shuffle=True,
                        num_workers=4,
                        # pin_memory=True,
                        collate_fn=my_collate
                        )
    loader_val_violence = DataLoader(dataset_val_violence,
                        batch_size=config.train_batch,
                        shuffle=True,
                        num_workers=4,
                        # pin_memory=True,
                        collate_fn=my_collate
                        )
   
    ################## Full Detector ########################
    
    #
    from models.violence_detector import ViolenceDetectorBinary
    if config.model == 'densenet_lean_roi':
        model, params = VioNet_densenet_lean_roi(config, config.pretrained_model)
    elif config.model == 'i3d+roi+fc':
        model, params = ViolenceDetector_model(config, device, config.pretrained_model)
    elif config.model == 'i3d+roi+i3d':
        model, params = VioNet_I3D_Roi(config, device, config.pretrained_model)
    elif config.model == 'i3d+roi+binary':
        model = ViolenceDetectorBinary(
            freeze=config.freeze).to(device)
        params = model.parameters()

    exp_config_log = "SpTmpDetector_{}_model({})_head({})_stream({})_cv({})_epochs({})_tubes({})_tub_sampl_rand({})_optimizer({})_lr({})_note({})".format(config.dataset,
                                                                config.model,
                                                                config.head,
                                                                config.input_type,
                                                                config.num_cv,
                                                                config.num_epoch,
                                                                config.num_tubes,
                                                                config.tube_sampling_random,
                                                                config.optimizer,
                                                                config.learning_rate,
                                                                config.additional_info)
    
    h_p = HOME_DRIVE if config.home_path==HOME_COLAB else config.home_path
    tsb_path_folder = os.path.join(h_p, PATH_TENSORBOARD, exp_config_log)
    chk_path_folder = os.path.join(h_p, PATH_CHECKPOINT, exp_config_log)

    for p in [tsb_path_folder, chk_path_folder]:
        if not os.path.exists(p):
            os.makedirs(p)
    # print('tensorboard dir:', tsb_path)                                                
    writer = SummaryWriter(tsb_path_folder)

    if config.optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(params, lr=config.learning_rate, eps=1e-8)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=params,
                                    lr=config.learning_rate,
                                    momentum=0.5,
                                    weight_decay=1e-3)
    
    if config.head == REGRESSION:
        # criterion = nn.BCELoss().to(device)
        # criterion = nn.BCEWithLogitsLoss().to(device)
        criterion = MIL
    elif config.head == BINARY:
        criterion = nn.CrossEntropyLoss().to(config.device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           verbose=True,
                                                           factor=config.factor,
                                                           min_lr=config.min_lr)
    from utils import AverageMeter
    # from epoch import calculate_accuracy_2

    start_epoch = 0
    ##Restore training
    if config.restore_training:
        model, optimizer, epochs, last_epoch, last_loss = load_checkpoint(model, config.device, optimizer, config.checkpoint_path)
        start_epoch = last_epoch+1
        # config.num_epoch = epochs
    
    def train(_epoch, _model, _criterion, _optimizer):
        print('training at epoch: {}'.format(epoch))
        _model.train()
        losses = AverageMeter()
        accuracies = AverageMeter()
        for i, data in enumerate(zip(loader_train_violence, loader_train_nonviolence)):
            video_images = torch.cat([data[0][1], data[1][1]], dim=0).to(device)
            boxes = torch.cat([data[0][0], data[1][0]], dim=0).to(device)
            labels = torch.cat([data[0][2], data[1][2]], dim=0).to(device)
            # print('video_images: ', video_images.size())
            # print('num_tubes: ', config.num_tubes)
            # print('boxes: ', boxes.size())
            # print('labels: ', labels, labels.size())

            # zero the parameter gradients
            optimizer.zero_grad()
            #predict
            outs = _model(video_images, boxes, config.num_tubes)
            # print('outs: ', outs, outs.size())
            #loss
            loss = _criterion(outs,labels)
            #accuracy
            if config.head == REGRESSION:
                acc = get_accuracy(outs, labels)
            else:
                acc = calculate_accuracy_2(outs,labels)
            # meter
            # print('len(video_images): ', len(video_images), ' video_images.size(0):',video_images.size(0), ' preds.shape[0]:', preds.shape[0])
            losses.update(loss.item(), outs.shape[0])
            accuracies.update(acc, outs.shape[0])
            # backward + optimize
            loss.backward()
            _optimizer.step()
        train_loss = losses.avg
        train_acc = accuracies.avg
        print(
            'Epoch: [{}]\t'
            'Loss(train): {loss.avg:.4f}\t'
            'Acc(train): {acc.avg:.3f}'.format(_epoch, loss=losses, acc=accuracies)
        )
        return train_loss, train_acc


    def val(_epoch, _model, _criterion):
        print('validation at epoch: {}'.format(epoch))
        # set model to evaluate mode
        _model.eval()
        # meters
        losses = AverageMeter()
        accuracies = AverageMeter()
        for i, data in enumerate(zip(loader_val_violence, loader_val_nonviolence)):
            video_images = torch.cat([data[0][1], data[1][1]], dim=0).to(device)
            boxes = torch.cat([data[0][0], data[1][0]], dim=0).to(device)
            labels = torch.cat([data[0][2], data[1][2]], dim=0).to(device)
            # no need to track grad in eval mode
            with torch.no_grad():
                outputs = _model(video_images, boxes, config.num_tubes)
                loss = _criterion(outputs, labels)
                
                if config.head == REGRESSION:
                    acc = get_accuracy(outputs, labels)
                else:
                    acc = calculate_accuracy_2(outputs,labels)

            losses.update(loss.item(), outputs.shape[0])
            accuracies.update(acc, outputs.shape[0])
        val_loss = losses.avg
        val_acc = accuracies.avg
        print(
            'Epoch: [{}]\t'
            'Loss(val): {loss:.4f}\t'
            'Acc(val): {acc:.3f}'.format(_epoch, loss=val_loss, acc=val_acc)
        )
        

        return val_loss, val_acc

    for epoch in range(start_epoch, config.num_epoch):
        # epoch = last_epoch+i
        train_loss, train_acc = train(epoch, model, criterion, optimizer)
        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('training accuracy', train_acc, epoch)
        
        val_loss, val_acc = val(epoch, model, criterion)
        scheduler.step(val_loss)
        writer.add_scalar('validation loss', val_loss, epoch)
        writer.add_scalar('validation accuracy', val_acc, epoch)

        if (epoch+1)%config.save_every == 0:
            save_checkpoint(model, config.num_epoch, epoch, optimizer,train_loss, os.path.join(chk_path_folder,"save_at_epoch-"+str(epoch)+".chk"))


def MIL_training(config: Config):
    device = config.device
    make_dataset_nonviolence = load_make_dataset(
        config.dataset, 
        train=True,
        cv_split=config.num_cv, 
        home_path=config.home_path,
        category=0
        )
    make_dataset_violence = load_make_dataset(
        config.dataset, 
        train=True,
        cv_split=config.num_cv, 
        home_path=config.home_path,
        category=1
        )
    spatial_t = DefaultTrasformations(model_name=config.model, size=224, train=True)
    dataset_train_nonviolence = TubeDataset(frames_per_tube=16, 
                            min_frames_per_tube=8,
                            make_function=make_dataset_nonviolence,
                            spatial_transform=spatial_t(),
                            max_num_tubes=config.num_tubes,
                            train=True,
                            dataset=config.dataset,
                            input_type=config.input_type,
                            random=config.tube_sampling_random)
    dataset_train_violence = TubeDataset(frames_per_tube=16, 
                            min_frames_per_tube=8,
                            make_function=make_dataset_violence,
                            spatial_transform=spatial_t(),
                            max_num_tubes=config.num_tubes,
                            train=True,
                            dataset=config.dataset,
                            input_type=config.input_type,
                            random=config.tube_sampling_random)
    loader_nonviolence = DataLoader(dataset_train_nonviolence,
                        batch_size=config.train_batch,
                        shuffle=True,
                        num_workers=4,
                        # pin_memory=True,
                        collate_fn=my_collate
                        )
    loader_violence = DataLoader(dataset_train_violence,
                        batch_size=config.train_batch,
                        shuffle=True,
                        num_workers=4,
                        # pin_memory=True,
                        collate_fn=my_collate
                        )

    #validation
    
   
    ################## Full Detector ########################
    
    model, params = ViolenceDetector_model(config, device, config.pretrained_model)

    exp_config_log = "SpTmpDetector_{}_model({})_head({})_stream({})_cv({})_epochs({})_tubes({})_tub_sampl_rand({})_optimizer({})_lr({})_note({})".format(config.dataset,
                                                                config.model,
                                                                config.head,
                                                                config.input_type,
                                                                config.num_cv,
                                                                config.num_epoch,
                                                                config.num_tubes,
                                                                config.tube_sampling_random,
                                                                config.optimizer,
                                                                config.learning_rate,
                                                                config.additional_info)
    
    h_p = HOME_DRIVE if config.home_path==HOME_COLAB else config.home_path
    tsb_path_folder = os.path.join(h_p, PATH_TENSORBOARD, exp_config_log)
    chk_path_folder = os.path.join(h_p, PATH_CHECKPOINT, exp_config_log)

    for p in [tsb_path_folder, chk_path_folder]:
        if not os.path.exists(p):
            os.makedirs(p)                                               
    writer = SummaryWriter(tsb_path_folder)

    if config.optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(params, lr=config.learning_rate, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           verbose=True,
                                                           factor=config.factor,
                                                           min_lr=config.min_lr)   
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=params,
                                    lr=config.learning_rate,
                                    momentum=0.5,
                                    weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           verbose=True,
                                                           factor=config.factor,
                                                           min_lr=config.min_lr)
    elif config.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr= config.learning_rate, weight_decay=0.0010000000474974513)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
    
    
    criterion = MIL
    
    from utils import AverageMeter

    start_epoch = 0
    ##Restore training
    if config.restore_training:
        model, optimizer, epochs, last_epoch, last_loss = load_checkpoint(model, config.device, optimizer, config.checkpoint_path)
        start_epoch = last_epoch+1
        # config.num_epoch = epochs

    for epoch in range(start_epoch, config.num_epoch):
        print('training at epoch: {}'.format(epoch))
        model.train()
        losses = AverageMeter()
        accuracies = AverageMeter()
        train_loss = 0
        for i, data in enumerate(zip(loader_violence, loader_nonviolence)):
            # print('iiiiiiiii :' , i+1)
            # print('=============== violence batch')
            boxes, video_images, labels, num_tubes, paths = data[0]
            boxes, video_images = boxes.to(device), video_images.to(device)
            labels = labels.float().to(device) if config.head == REGRESSION else labels.to(device)

            # print('video_images: ', video_images.size())
            # print('num_tubes: ', config.num_tubes)
            # print('boxes: ', boxes.size())

            # print('labels: ', labels)

            # print('=============== nonviolence batch')
            boxes, video_images, labels, num_tubes, paths = data[1]
            boxes, video_images = boxes.to(device), video_images.to(device)
            labels = labels.float().to(device) if config.head == REGRESSION else labels.to(device)

            # print('video_images: ', video_images.size())
            # print('num_tubes: ', config.num_tubes)
            # print('boxes: ', boxes.size())

            # print('labels: ', labels)

            video_images = torch.cat([data[0][1], data[1][1]], dim=0).to(device)
            boxes = torch.cat([data[0][0], data[1][0]], dim=0).to(device)

            # print('video_images cat: ', video_images.size())
            # print('boxes cat: ', boxes.size())

            # zero the parameter gradients
            optimizer.zero_grad()
            #predict
            outs = model(video_images, boxes, config.num_tubes)
            # print('outs: ', outs.size())
            #loss
            loss = criterion(outs,config.train_batch)
            train_loss += loss.item()
            # backward + optimize
            loss.backward()
            optimizer.step()
        train_loss = train_loss/min(len(loader_violence), len(loader_nonviolence))
        print(
            'Epoch: [{}]\t'
            'Loss(train): {loss:.4f}\t'.format(epoch, loss=train_loss)
        )
        scheduler.step(train_loss)
        writer.add_scalar('training loss', train_loss, epoch)

        if (epoch+1)%config.save_every == 0:
            save_checkpoint(model, config.num_epoch, epoch, optimizer,train_loss, os.path.join(chk_path_folder,"save_at_epoch-"+str(epoch)+".chk"))



def get_accuracy(y_prob, y_true):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()

    # print('y_true:', y_true, y_true.size())
    # print('y_prob:', y_prob, y_prob.size())

    y_prob = y_prob >= 0.5
    # print('(y_true == y_prob):', (y_true == y_prob))
    return (y_true == y_prob).sum().item() / y_true.size(0)


if __name__=='__main__':
    config = Config(
        model='i3d+roi+binary',#'i3d-roi',i3d+roi+fc
        head=BINARY,
        dataset=RWF_DATASET,
        num_cv=1,
        input_type='rgb',
        device=get_torch_device(),
        num_epoch=100,
        optimizer='SGD',
        learning_rate=0.001, #0.001 for adagrad
        train_batch=8,
        val_batch=8,
        num_tubes=4,
        tube_sampling_random=True,
        save_every=10,
        freeze=True,
        additional_info='usingbalanceddatasets3',
        home_path=HOME_UBUNTU
    )
    # config.pretrained_model = "/content/DATASETS/Pretrained_Models/DenseNetLean_Kinetics.pth"
    # config.pretrained_model='/media/david/datos/Violence DATA/VioNet_weights/pytorch_i3d/rgb_imagenet.pt'
    # config.pretrained_model = '/media/david/datos/Violence DATA/VioNet_pth/rwf_trained/save_at_epoch-127.chk'
    # config.restore_training = True
    # config.checkpoint_path = os.path.join(config.home_path,
    #                                       PATH_CHECKPOINT,
    #                                       'SpTmpDetector_rwf-2000_model(binary)_stream(rgb)_cv(1)_epochs(200)_note(restorefrom97epoch)',
    #                                       'rwf_trained/save_at_epoch-127.chk')

    # main(config)
    main_2(config)
    # MIL_training(config)
    # extract_features(config, output_folder='/media/david/datos/Violence DATA/i3d-FeatureMaps/rwf')
    # load_features(config)