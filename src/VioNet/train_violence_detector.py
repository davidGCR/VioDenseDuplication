
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
from customdatasets.tube_dataset import TubeDataset, my_collate, OneVideoTubeDataset, TubeFeaturesDataset
from torch.utils.data import DataLoader, dataset

from config import Config
from model import ViolenceDetector_model
from models.anomaly_detector import custom_objective, RegularizedLoss
from epoch import train_regressor
from utils import Log, save_checkpoint, load_checkpoint
from utils import get_torch_device
from feature_writer2 import FeaturesWriter
# device = get_torch_device()

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
    make_dataset = MakeRWF2000(root='/media/david/datos/Violence DATA/RWF-2000/frames',#'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames', 
                                train=False,
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
                        batch_size=4,
                        shuffle=True,
                        num_workers=4,
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
    model.eval()
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

    from models.anomaly_detector import custom_objective
    from torch import nn
    bceloss = nn.BCELoss()

    ##iterate over dataset
    for i, data in enumerate(loader): #iterate over batches of videos
        # path, label, annotation,frames_names, boxes, video_images = data
        boxes, video_images, labels = data
        # boxes, video_images, labels = boxes.to(device), video_images.to(device), labels.to(device)
        print('_____ {} ______'.format(i+1))

        if not boxes:
            print('No boxes')
            continue
        
        # print('boxes: ', type(boxes), len(boxes))#, '-boxes[0]: ', boxes[0].size())
        # print('video_images: ', type(video_images), len(video_images), '-video_images[0]: ', video_images[0].size())
        labels = torch.tensor(labels).float().to(device)
        print('labels: ', type(labels), len(labels), '-labels: ', labels)
        
        
        scores = []
        for j in range(len(video_images)): # iterate over videos into a batch
            video_images[j] = video_images[j].to(device)
            boxes[j] = boxes[j].to(device)
            # labels[i] = labels[i].to(device)

            # print('boxes[{}]: '.format(i), boxes[i], boxes[i].size())
            print('video_images[{}]: '.format(j), video_images[j].size())
            # print('labels[{}]: '.format(i), labels[i])

            # out = model(video_images[i])
            # print('out feature: ', out.size())

            # y_pred = model(video_images[i], boxes[i])
            # roi_out = roi_op(video_images[i])

            out = model(video_images[j], boxes[j]) #get score of video tubes
            print('out[{}]: {}, {}'.format(j, out, out.size()))

            # get the max score for each video
            instance_max_score = out.max(dim=0)[0]
            scores.append(instance_max_score.item())
            # anomal_segments_scores_maxes = out.max(dim=-1)[0]

            # print("--instance_max_score:", instance_max_score)
        scores = torch.tensor(scores).float().to(device)
        print('scores: {} - {}, dtype:{}'.format(scores, scores.size(), scores.type()))
        loss = bceloss(scores,labels)
        print('loss: ', loss)

if __name__=='__main__':
    config = Config(
        model='',
        dataset='',
        device=get_torch_device()
    )
    main(config)
    # extract_features(config, output_folder='/media/david/datos/Violence DATA/i3d-FeatureMaps/rwf')
    # load_features(config)