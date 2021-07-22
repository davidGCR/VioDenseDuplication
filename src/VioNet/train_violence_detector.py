
from os import path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch._C import device
from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision.transforms.transforms import Resize
import torchvision.transforms as transforms
import os

#data
from customdatasets.make_dataset import MakeRWF2000
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

def main(config: Config):
    device = config.device
    #'/content/DATASETS/RWF-2000/frames'
    #'/content/DATASETS/ActionTubes/RWF-2000'
    make_dataset = MakeRWF2000(root='/media/david/datos/Violence DATA/RWF-2000/frames',#'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames', 
                                train=True,
                                path_annotations='/media/david/datos/Violence DATA/ActionTubes/RWF-2000')#'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/Tubes/RWF-2000')
    dataset = TubeDataset(frames_per_tube=16, 
                            min_frames_per_tube=8, 
                            make_function=make_dataset,
                            spatial_transform=transforms.Compose([
                                # transforms.CenterCrop(224),
                                # transforms.Resize(256),
                                transforms.ToTensor()
                            ]),
                            max_num_tubes=4)
    loader = DataLoader(dataset,
                        batch_size=2,
                        shuffle=True,
                        num_workers=1,
                        # pin_memory=True,
                        collate_fn=my_collate
                        )

    ################## Head Detector ########################
    # from models.roi_extractor_3d import SingleRoIExtractor3D
    # from models.anomaly_detector import AnomalyDetector
    # roi_op = SingleRoIExtractor3D(roi_layer_type='RoIAlign',
    #                             featmap_stride=16,
    #                             output_size=8,    
    #                             with_temporal_pool=True)
    # model = AnomalyDetector()

    # from models.violence_detector import RoiHead
    # head = RoiHead()
    # head.to(device)

    ################## Full Detector ########################

    model, params = ViolenceDetector_model(config, device)
    # torch.backends.cudnn.enabled = False
    # model.eval()
    # print(model)
    # optimizer = torch.optim.Adadelta(params, lr=config.learning_rate, eps=1e-8)

    # criterion = RegularizedLoss(model, custom_objective)
    
    ################## Feature Extractor ########################
    # from models.i3d import InceptionI3d
    # model = InceptionI3d(2, in_channels=3, final_endpoint='Mixed_4e').to(device)
    # load_model_path = '/media/david/datos/Violence DATA/VioNet_weights/pytorch_i3d/rgb_imagenet.pt'
    # state_dict = torch.load(load_model_path)
    # model.load_state_dict(state_dict,  strict=False) 
    # model.eval()

    
    optimizer = torch.optim.Adadelta(params, lr=config.learning_rate, eps=1e-8)
    # optimizer = torch.optim.SGD(params=params,
    #                             lr=learning_rate,
    #                             momentum=momentum,
    #                             weight_decay=weight_decay)
    criterion = nn.BCELoss()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                        verbose=True,
    #                                                        factor=config.factor,
    #                                                        min_lr=config.min_lr)
    from utils import AverageMeter
    from epoch import calculate_accuracy_2

    # for epoch in range(config.num_epoch):
    #     model.train()
    #     losses = AverageMeter()
    #     accuracies = AverageMeter()
    #     for i, (boxes, video_images, labels) in enumerate(loader):
    #         labels = torch.tensor(labels).float().to(device)
    #         if not boxes:
    #             # print('No boxes')
    #             continue
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #         scores = []
    #         for j in range(len(video_images)): # iterate over videos into a batch
    #             video_images[j] = video_images[j].to(device)
    #             boxes[j] = boxes[j].to(device)
    #             # print('video_images[{}]: '.format(j), video_images[j].size())
    #             out = model(video_images[j], boxes[j]) #get score of video tubes
    #             # print('out[{}]: {}, {}'.format(j, out, out.size()))
    #             # get the max score for each video
    #             instance_max_score = out.max(dim=0)[0]
    #             scores.append(instance_max_score.item())
                
    #         scores = torch.tensor(scores).float().to(device)
    #         # print('scores: {} - {}, dtype:{}'.format(scores, scores.size(), scores.type()))
    #         loss = criterion(scores,labels)

    #         preds = np.round(scores.cpu().numpy())
    #         acc = (preds == labels.cpu().numpy()).sum() / preds.shape[0]
    #         # calculate_accuracy_2(scores, labels)

    #         # meter
    #         losses.update(loss.item(), len(video_images))
    #         accuracies.update(acc, len(video_images))

    #         # backward + optimize
    #         loss.backward()
    #         optimizer.step()

    #         # meter
    #         # batch_time.update(time.time() - end_time)
    #         # end_time = time.time()

    #         print(
    #             'Epoch: [{0}][{1}/{2}]\t'
    #             # 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #             # 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
    #             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #             'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
    #                 epoch,
    #                 i + 1,
    #                 len(loader),
    #                 # batch_time=batch_time,
    #                 # data_time=data_time,
    #                 loss=losses,
    #                 acc=accuracies
    #             )
    #         )


    ##iterate over dataset
    for i, data in enumerate(loader): #iterate over batches of videos
        # path, label, annotation,frames_names, boxes, video_images = data
        boxes, video_images, labels, num_tubes, paths = data
        boxes, video_images, labels = boxes.to(device), video_images.to(device), labels.to(device)
        print('_____ {} ______'.format(i+1))
        print('paths: ', paths)
        if boxes.size(0)==0:
            print('No boxes')
            continue
        
        print('boxes: ', type(boxes), boxes.size())#, '-boxes[0]: ', boxes[0].size())
        print('video_images: ', type(video_images), video_images.size())
        print('labels: ', type(labels), labels.size())
        print('num_tubes: ', num_tubes)
        
        # labels = torch.tensor(labels).float().to(device)
        # print('labels: ', type(labels), len(labels), '-labels: ', labels)
        
        # out = model(video_images, boxes)
        # print('out: ', out.size())
        # scores = []
        # for j in range(len(video_images)): # iterate over videos into a batch
        #     video_images[j] = video_images[j].to(device)
        #     boxes[j] = boxes[j].to(device)
        #     # print('video_images[{}]: '.format(j), video_images[j].size())
        #     out = model(video_images[j], boxes[j]) #get score of video tubes
        #     # print('out[{}]: {}, {}'.format(j, out, out.size()))
        #     # get the max score for each video
        #     instance_max_score = out.max(dim=0)[0]
        #     scores.append(instance_max_score.item())
            
        # scores = torch.tensor(scores).float().to(device)
        # # print('scores: {} - {}, dtype:{}'.format(scores, scores.size(), scores.type()))
        # loss = criterion(scores,labels)
        # # print('loss: ', loss)

if __name__=='__main__':
    config = Config(
        model='',
        dataset='',
        device=get_torch_device(),
        num_epoch=20,
        learning_rate=0.001
    )
    main(config)
    # extract_features(config, output_folder='/media/david/datos/Violence DATA/i3d-FeatureMaps/rwf')
    # load_features(config)