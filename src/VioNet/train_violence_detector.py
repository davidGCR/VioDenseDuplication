
from os import path
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
from model import ViolenceDetector_model
from models.anomaly_detector import custom_objective, RegularizedLoss
from epoch import train_regressor
from utils import Log, save_checkpoint, load_checkpoint
from utils import get_torch_device
from feature_writer2 import FeaturesWriter
from models.anomaly_detector import custom_objective
from torch import nn
from epoch import train, val
import numpy as np
from global_var import *

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

def load_make_dataset(dataset_name, train=True, cv_split=1):
    if dataset_name == RWF_DATASET:
        make_dataset = MakeRWF2000(root='/media/david/datos/Violence DATA/RWF-2000/frames',#'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames', 
                                train=train,
                                path_annotations='/media/david/datos/Violence DATA/ActionTubes/RWF-2000')#'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/Tubes/RWF-2000')
    elif dataset_name == HOCKEY_DATASET:
        make_dataset = MakeHockeyDataset(root='/media/david/datos/Violence DATA/DATASETS/HockeyFightsDATASET/frames', 
                                        train=train,
                                        cv_split_annotation_path='/media/david/datos/Violence DATA/VioDB/hockey_jpg{}.json'.format(cv_split),
                                        path_annotations='/media/david/datos/Violence DATA/ActionTubes/hockey')
    return make_dataset


def main(config: Config):
    device = config.device
    #'/content/DATASETS/RWF-2000/frames'
    #'/content/DATASETS/ActionTubes/RWF-2000'

    make_dataset = load_make_dataset(config.dataset, train=True, cv_split=config.num_cv)
    
    dataset = TubeDataset(frames_per_tube=16, 
                            min_frames_per_tube=8,
                            make_function=make_dataset,
                            spatial_transform=transforms.Compose([
                                # transforms.CenterCrop(224),
                                transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize([38.756858/255, 3.88248729/255, 40.02898126/255], [110.6366688/255, 103.16065604/255, 96.29023126/255])
                            ]),
                            max_num_tubes=4,
                            train=True)
    loader = DataLoader(dataset,
                        batch_size=config.train_batch,
                        shuffle=True,
                        num_workers=4,
                        # pin_memory=True,
                        collate_fn=my_collate
                        )

    #validation
    val_make_dataset = load_make_dataset(config.dataset, train=False, cv_split=config.num_cv)
    val_dataset = TubeDataset(frames_per_tube=16, 
                            min_frames_per_tube=8, 
                            make_function=val_make_dataset,
                            spatial_transform=transforms.Compose([
                                # transforms.CenterCrop(224),
                                transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize([38.756858/255, 3.88248729/255, 40.02898126/255], [110.6366688/255, 103.16065604/255, 96.29023126/255])
                            ]),
                            max_num_tubes=4,
                            train=False)
    val_loader = DataLoader(val_dataset,
                        batch_size=config.val_batch,
                        shuffle=True,
                        num_workers=1,
                        # pin_memory=True,
                        collate_fn=my_collate
                        )
   
    ################## Full Detector ########################
    model, params = ViolenceDetector_model(config, device)
    exp_config_log = "SpTmpDetector_{}_cv({})_epochs({})_note({})".format(config.dataset,
                                                                config.num_cv,
                                                                config.num_epoch,
                                                                config.additional_info)
    tsb_path_folder = os.path.join(HOME_UBUNTU, PATH_TENSORBOARD, exp_config_log)
    chk_path_folder = os.path.join(HOME_UBUNTU, PATH_CHECKPOINT, exp_config_log)

    for p in [tsb_path_folder, chk_path_folder]:
        if not os.path.exists(p):
            os.mkdir(p)
    # print('tensorboard dir:', tsb_path)                                                
    writer = SummaryWriter(tsb_path_folder)
    
    optimizer = torch.optim.Adadelta(params, lr=config.learning_rate, eps=1e-8)
    # optimizer = torch.optim.SGD(params=params,
    #                             lr=config.learning_rate,
    #                             momentum=0.5,
    #                             weight_decay=1e-3)
    criterion = nn.BCELoss().to(device)
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
        config.num_epoch = epochs

    for epoch in range(start_epoch, config.num_epoch):
        # epoch = last_epoch+i
        print('training at epoch: {}'.format(epoch))
        model.train()
        losses = AverageMeter()
        accuracies = AverageMeter()
        for i, data in enumerate(loader):
            boxes, video_images, labels, num_tubes, paths = data
            boxes, video_images, labels = boxes.to(device), video_images.to(device), labels.float().to(device)
            # print('video_images: ', video_images.size())
            # print('num_tubes: ', num_tubes)
            # print('boxes: ', boxes, boxes.size())


            # zero the parameter gradients
            optimizer.zero_grad()
            #predict
            outs = model(video_images, boxes, num_tubes)
            #loss
            loss = criterion(outs,labels)
            #accuracy
            # preds = np.round(scores.detach().cpu().numpy())
            # acc = (preds == labels.cpu().numpy()).sum() / preds.shape[0]
            acc = get_accuracy(outs, labels)
            # meter
            # print('len(video_images): ', len(video_images), ' video_images.size(0):',video_images.size(0), ' preds.shape[0]:', preds.shape[0])
            losses.update(loss.item(), outs.shape[0])
            accuracies.update(acc, outs.shape[0])
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
                    len(loader),
                    # batch_time=batch_time,
                    # data_time=data_time,
                    loss=losses,
                    acc=accuracies
                )
            )
        train_loss = losses.avg

        writer.add_scalar('training loss', losses.avg, epoch)
        writer.add_scalar('training accuracy', accuracies.avg, epoch)
        
        print('validation at epoch: {}'.format(epoch))
        # set model to evaluate mode
        model.eval()
        # meters
        losses = AverageMeter()
        accuracies = AverageMeter()
        for _, data in enumerate(val_loader):
            boxes, video_images, labels, num_tubes, paths = data
            boxes, video_images, labels = boxes.to(device), video_images.to(device), labels.float().to(device)
            # no need to track grad in eval mode
            with torch.no_grad():
                outputs = model(video_images, boxes, num_tubes)
                loss = criterion(outputs, labels)
                # preds = np.round(outputs.detach().cpu().numpy())
                # acc = (preds == labels.cpu().numpy()).sum() / preds.shape[0]
                acc = get_accuracy(outputs, labels)

            losses.update(loss.item(), outputs.shape[0])
            accuracies.update(acc, outputs.shape[0])

        print(
            'Epoch: [{}]\t'
            'Loss(val): {loss.avg:.4f}\t'
            'Acc(val): {acc.avg:.3f}'.format(epoch, loss=losses, acc=accuracies)
        )
        scheduler.step(losses.avg)
        writer.add_scalar('validation loss', losses.avg, epoch)
        writer.add_scalar('validation accuracy', accuracies.avg, epoch)

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
        model='',
        dataset=RWF_DATASET,
        num_cv=1,
        device=get_torch_device(),
        num_epoch=1000,
        learning_rate=0.01,
        train_batch=4,
        val_batch=4,
        save_every=100,
        additional_info='0'
    )
    # config.restore_training = True
    # config.checkpoint_path = '/media/david/datos/Violence DATA/VioNet_pth/SpTmpDetector_hockey_cv(1)_epochs(1000)_note(-3)/save_at_epoch-1.chk'

    main(config)
    # extract_features(config, output_folder='/media/david/datos/Violence DATA/i3d-FeatureMaps/rwf')
    # load_features(config)