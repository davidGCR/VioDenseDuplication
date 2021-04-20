import sys
import os
from torchvision import transforms
import torchvision
import torch

g_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('main g_path:', g_path)
sys.path.insert(1, g_path)
# from data_loader import SegmentsCrop, DynamicImageIter
from VioNet.customdatasets.video_image_dataset import VideoImageDataset
from VioNet.customdatasets.make_dataset import MakeImageHMDB51, MakeRWF2000, MakeUCFCrime2LocalClips
from VioNet.utils import get_torch_device
from VioNet.config import Config
from VioNet.model import FeatureExtractor_ResnetXT, Feature_Extractor_S3D
from VioNet.transformations.temporal_transforms import DynamicImage
from VioNet.transformations.s3d_transform import s3d_transform
from VioNet.customdatasets.video_dataset import VideoDataset
from VioNet.utils import show_batch
from feature_writer import FeaturesWriter
import numpy as np

def extract_from_2d(config: Config, root, annotation_path, save_dir):
    network = FeatureExtractor_ResnetXT(config.device, config.pretrained_model)
    mean = [0.49778724, 0.49780366, 0.49776983]
    std = [0.09050678, 0.09017131, 0.0898702]
    size = config.sample_size

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
    tmp_transform = DynamicImage(output_type="pil")
    
    ############################################################
    #################### HMDB51 ################################
    ############################################################
    
    # make_function = MakeImageHMDB51(root=root,
    #                                 annotation_path=annotation_path,
    #                                 fold=config.num_cv,
    #                                 train=True)

    # data = VideoImageDataset(root="",
    #                                         frames_per_clip=config.sample_duration, 
    #                                         number_of_clips=config.number_of_clips, 
    #                                         make_function=make_function, 
    #                                         stride=config.stride, 
    #                                         overlap=config.overlap,
    #                                         position=config.position,
    #                                         temporal_transform=tmp_transform, 
    #                                         spatial_transform=data_transforms["val"],
    #                                         return_metadata=True)

    ############################################################
    #################### RWF-2000 ################################
    ############################################################
    
    # make_function = MakeRWF2000(root=root, train=True)

    # data = VideoImageDataset(root="",
    #                         frames_per_clip=config.sample_duration, 
    #                         number_of_clips=config.number_of_clips, 
    #                         make_function=make_function, 
    #                         stride=config.stride, 
    #                         overlap=config.overlap,
    #                         position=config.position,
    #                         temporal_transform=tmp_transform, 
    #                         spatial_transform=data_transforms["val"],
    #                         return_metadata=True)

    ############################################################
    ################ Video Clip Dataset ########################
    ############################################################
    # data = VideoDataset(clip_length=config.sample_duration,
    #                         frame_stride=config.stride,
    #                         frame_rate=25,
    #                         dataset_path= root,
    #                         temporal_transform=tmp_transform,
    #                         spatial_transform=data_transforms["val"])

    ############################################################
    ############### UCFCrime2LocalClips ########################
    ############################################################
    m = MakeUCFCrime2LocalClips(root=root)
    
    # mean=None
    # std=None
    data=VideoImageDataset(root="",
                        frames_per_clip=config.sample_duration, 
                        number_of_clips=config.number_of_clips, #this will return all segments
                        make_function=m, 
                        stride=config.stride, 
                        overlap=config.overlap, 
                        position="",
                        padding=False,
                        return_metadata=True,
                        temporal_transform=tmp_transform, 
                        spatial_transform=data_transforms['val'])

    data_iter = torch.utils.data.DataLoader(data,
                                            batch_size=config.val_batch,
                                            shuffle=False,
                                            num_workers=4,
                                            pin_memory=True)

    features_writer = FeaturesWriter(num_videos=data.video_count, num_segments=config.num_segments)

    with torch.no_grad():
        for inputs, labels, path in data_iter:
            inputs = torch.squeeze(inputs, dim=0) #remove batch dimension of 1
            print('video:', path[0])
            print('inputs:', inputs.size())

            outputs = network(inputs.to(device)).detach().cpu().numpy()
            for idx  in range(inputs.size()[0]):
                clip_idx = idx
                dir = "abnormal" if labels.item() == 1 else "normal"
                _, file = os.path.split(path[0])
                if os.path.isdir(os.path.join(save_dir,dir,file+'.txt')):
                    print("Already done!!! {}/{}".format(dir,file+'.txt'))
                    break
                dir = os.path.join(save_dir, dir)
                features_writer.write(feature=outputs[idx],
                                        video_name=file,
                                        idx=clip_idx,
                                        dir=dir)
    features_writer.dump()

    #### Use tthis whith VideoClipDataset
    # with torch.no_grad():
    #     for inputs, labels, metadata in data_iter:
    #         # print('inputs:', inputs.size())
    #         # print('dynImgs[0]:', dynamic_images[0].size())
    #         # print('metadata:', len(metadata), type(metadata))
    #         # print('labels:', labels)
            
    #         # video_batch = dynamic_images.squeeze()
    #         outputs = network(inputs.to(device)).detach().cpu().numpy() #(13, 2048)
    #         # print('outputs:', outputs.shape)
    #         for idx  in range(inputs.size()[0]):
    #             clip_idx = metadata[0][idx].item()
    #             dir = metadata[1][idx]
    #             file = metadata[2][idx]

    #             if os.path.isdir(os.path.join(save_dir,dir,file+'.txt')):
    #                 print("Already done!!! {}/{}".format(dir,file+'.txt'))
    #                 break

    #             # print(clip_idx, dir, file)
    #             dir = os.path.join(save_dir, dir)
    #             # print("---dir:", dir)
    #             features_writer.write(feature=outputs[idx],
    #                                     video_name=file,
    #                                     idx=clip_idx,
    #                                     dir=dir)

    # features_writer.dump()

    ####################################
            # print('outputs:', outputs.shape)
            ##save to txt
            # dir, video_name = os.path.split(video_paths[0])
            # _, folder = os.path.split(dir)
            # txt_path = os.path.join(args.save_dir, folder, f"{video_name}.txt")
            # data = dict()
            # for i in outputs.shape[0]:
            # 	data[i] = list(outputs[i])
            # print("Dumping {}".format(txt_path))

            # with open(txt_path, 'w') as fp:
            # 	for idx in range(outputs.shape[0]):
            # 		d = outputs[idx]
            # 		d = [str(x) for x in d]
            # 		fp.write(' '.join(d) + '\n')

def extract_from_s3d(config: Config, root, save_dir):
    network = Feature_Extractor_S3D(config)
    data = VideoDataset(clip_length=config.sample_duration,
                            frame_stride=config.stride,
                            frame_rate=25,
                            dataset_path= root,
                            temporal_transform=None,
                            spatial_transform=s3d_transform)

    data_iter = torch.utils.data.DataLoader(data,
                                            batch_size=config.val_batch,
                                            shuffle=False,
                                            num_workers=4,
                                            pin_memory=True)

    features_writer = FeaturesWriter(num_videos=data.video_count,avg_segments=False)

    with torch.no_grad():
        for inputs, labels, metadata in data_iter:
            # print('inputs:', inputs.size(), inputs.dtype)
            # print('metadata:', metadata)
            # print('labels:', labels)
            
            outputs = network(inputs.to(device)).detach().cpu().numpy() #(13, 2048)
            # print('outputs:', outputs.shape)
            for idx  in range(inputs.size()[0]):
                clip_idx = metadata[0][idx].item()
                dir = metadata[1][idx]
                file = metadata[2][idx]

                if os.path.isdir(os.path.join(save_dir,dir,file+'.txt')):
                    print("Already done!!! {}/{}".format(dir,file+'.txt'))
                    break

                # print(clip_idx, dir, file)
                dir = os.path.join(save_dir, dir)
                # print("---dir:", dir)
                features_writer.write(feature=outputs[idx],
                                        video_name=file,
                                        idx=clip_idx,
                                        dir=dir)

    features_writer.dump()


if __name__ == "__main__":
    device = get_torch_device()
    config = Config(
        model='resnetXT',  # c3d, convlstm, densenet, densenet_lean, resnet50, densenet2D, resnetXT
        dataset='ucfcrime2localClips',
        device=device,
        val_batch=1,
        input_mode='dynamic-images',
        sample_size=(224,224),
        sample_duration=10,
        number_of_clips=0, #0 means return al segments
        stride=1,
        pretrained_model='/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/src/MyTrainedModels/resnetXT_fps10_hmdb511_52_0.3863_2.666646_SDI.pth',#'/Users/davidchoqueluqueroman/Documents/CODIGOS/S3D/S3D_kinetics400.pt',
        num_cv=1,
        num_segments=32 #0 means no average the features
    )
    # config.pretrained_model = '/Users/davidchoqueluqueroman/Documents/CODIGOS/AVSS2019/src/MyTrainedModels/resnetXT_fps10_hmdb511_52_0.3863_2.666646_SDI.pth'
    # root='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime'#'/Users/davidchoqueluqueroman/Documents/CODIGOS/DATASETS_Local/hmdb51/frames'
    # root='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/videos'
    # root='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/videos'#'/Volumes/TOSHIBA EXT/DATASET/RWF-2000/train'

    root = ('/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips',
            '/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/frames')
    save_dir = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/features_from({})_input({})_frames({})_num_segments({})'.format(config.dataset,config.input_mode, config.sample_duration, config.num_segments)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    annotation_path = None
    # annotation_path='/Users/davidchoqueluqueroman/Documents/CODIGOS/DATASETS_Local/hmdb51/testTrainMulti_7030_splits'
    # root='/content/DATASETS/HMDB51/frames'
    # annotation_path='/content/drive/MyDrive/VIOLENCE DATA/HMDB51/testTrainMulti_7030_splits'
    
    if config.input_mode == 'rgb':
        extract_from_s3d(config, root, save_dir)
    elif config.input_mode == 'dynamic-images':
        extract_from_2d(config, root, annotation_path, save_dir)
    