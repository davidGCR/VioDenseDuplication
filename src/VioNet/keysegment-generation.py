import os
from models.models2D import ResNet, Densenet2D, FeatureExtractorResNet, FeatureExtractorResNextTempPool
# from models.anomaly_detector import AnomalyDetector

from dataset import VioDB, ProtestDatasetEval, OneVideoFolderDataset
from torch.utils.data import DataLoader
import tqdm
import pandas as pd
import numpy as np
import torch
import argparse
import torchvision.transforms as transforms
from transformations.spatial_transforms import Compose, ToTensor, Normalize, GroupScaleCenterCrop
from transformations.temporal_transforms import CenterCrop, SequentialCrop
from transformations.target_transforms import Label
from transformations.s3d_transform import s3d_transform
from config import Config
from global_var import FEAT_EXT_RESNET, FEAT_EXT_RESNEXT, FEAT_EXT_RESNEXT_S3D, FEAT_EXT_S3D
from model import Feature_Extractor_S3D, FeatureExtractor_ResnetXT, AnomalyDetector_model

g_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sm = torch.nn.Softmax()

def eval_one_dir_classifier(img_dir, model):
  
    """
    return model output of all the images in a directory
    """
    
    model.eval()
    # make dataloader
    sample_size = 224
    crop_method = GroupScaleCenterCrop(size=sample_size)
    norm = Normalize([0.49778724, 0.49780366, 0.49776983], [0.09050678, 0.09017131, 0.0898702 ])
    dataset = "rwf-2000"
    cv = 1
    sample_duration = 5
    stride = 1
    input_mode = "dynamic-images"
    overlap = 0.5
    batch_size = 8

    spatial_transform = Compose([crop_method, ToTensor(), norm])
    target_transform = Label()
    temporal_transform = SequentialCrop(size=sample_duration, stride=stride, input_type=input_mode, overlap=overlap)

    val_dataset = RwfDatasetEval(img_dir, spatial_transform, temporal_transform)

    data_loader = DataLoader(val_dataset,
                            num_workers = 4,
                            batch_size = batch_size,
                            shuffle=False)

    predictions = []
    imgpaths = []

    # n_imgs = len(os.listdir(img_dir))
    # with tqdm(total=n_imgs) as pbar:
    # print('len(dataloader.dataset): ', len(data_loader.dataset))
    for i, sample in enumerate(data_loader):
        input, segment = sample
        # listToStr = '-'.join([str(elem.cpu().data.numpy()) for elem in segment])
        # print('segment:', segment)
        input_var = input.to(device)
        output = model(input_var)
        _, pred = output.topk(1, 1, True)
        probabilities = sm(output) 

        # print("pred.size(): ",pred.size(), pred)
        # print("probabilities:", probabilities.size(), probabilities)
        result = np.concatenate((pred.cpu().data.numpy(), probabilities.cpu().data.numpy()), axis=1)
        # print("result:", result.shape, result)
        predictions.append(result)
        imgpaths += segment

    # print('len(imgpaths):', len(imgpaths))

    df = pd.DataFrame(np.zeros((len(data_loader.dataset), 4)))
    df.columns = ["imgpath", "pred", "no_violence", "violence"]
    df['imgpath'] = imgpaths
   
    df.iloc[:,1:] = np.concatenate(predictions)
    # df.sort_values(by = 'imgpath', inplace=True)
    # print(df.head())
    return df

def eval_one_dir_an(config: Config, img_dir, feature_extractor, spatial_transform, temporal_transform, anomaly_detector):
  
    """
    return model output of all the images in a directory
    """
    
    if isinstance(feature_extractor, tuple):
        feature_extractor[0].eval()
        feature_extractor[1].eval()
    else:
        feature_extractor.eval()
    anomaly_detector.eval()
    # make dataloader

    val_dataset = OneVideoFolderDataset(img_dir, config.dataset, spatial_transform, temporal_transform)

    data_loader = DataLoader(val_dataset,
                            num_workers = 4,
                            batch_size = config.val_batch,
                            shuffle=False)

    predictions = []
    imgpaths = []
    for i, sample in enumerate(data_loader):
        input, segment, images = sample

        input = torch.squeeze(input, 2)
        # print('segment:', segment)
        # print('input:', input.size())
        # print('images:', images.size(), images.dtype)

        input_var = input.to(device)
        if isinstance(feature_extractor, tuple):
            feature_1 = feature_extractor[0](input_var)
            feature_2 = feature_extractor[1](images.to(device))
            feature = torch.cat([feature_1, feature_2], dim=1)
        else:
            feature = feature_extractor(input_var)
        # print('feature:', feature.size())
        score = anomaly_detector(feature)

        result = score.cpu().data.numpy()
        predictions.append(result)
        imgpaths += segment

        # print("feature: ", feature.size())
        # print("score: ", score.size())
        # _, pred = output.topk(1, 1, True)
        # probabilities = sm(output) 


    df = pd.DataFrame(np.zeros((len(data_loader.dataset), 2)))
    df.columns = ["imgpath", "score"]
    df['imgpath'] = imgpaths
    df.iloc[:,1:] = np.concatenate(predictions)

    # df.sort_values(by = 'imgpath', inplace=True)
    # print(df.head())
    return df
    # return 0

def load_classifier():
    model = Densenet2D()
    if device==torch.device('cpu'):
        state_dict = torch.load(args.model, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(args.model)
    model.load_state_dict(state_dict)
    return model

def load_feature_extractor(config: Config, source):
    # if source == FEAT_EXT_RESNET:
    #     model = FeatureExtractorResNet()
    # elif source == FEAT_EXT_RESNEXT:
    #     model = FeatureExtractorResNextTempPool()

    max_segments = 7 if config.dataset=="rwf-2000" else 4
    if source == FEAT_EXT_RESNEXT_S3D:
        model_1, model_2 = FeatureExtractor_ResnetXT(config.device, config.pretrained_fe[0]), Feature_Extractor_S3D(config.device, config.pretrained_fe[1])
        model = (model_1, model_2)

        crop_method = GroupScaleCenterCrop(size=config.sample_size)
        norm = Normalize([0.49778724, 0.49780366, 0.49776983], [0.09050678, 0.09017131, 0.0898702 ])
        dataset = config.dataset

        spatial_transform_1 = Compose([crop_method, ToTensor(), norm])
        temporal_transform = SequentialCrop(size=config.sample_duration, stride=config.stride, overlap=config.overlap, max_segments=max_segments)
        spatial_transform = (spatial_transform_1, s3d_transform)
    elif source == FEAT_EXT_RESNEXT:
        model = FeatureExtractor_ResnetXT(config.device, config.pretrained_fe)
        # crop_method = GroupScaleCenterCrop(size=config.sample_size)
        # norm = Normalize([0.49778724, 0.49780366, 0.49776983], [0.09050678, 0.09017131, 0.0898702 ])
        mean = [0.49778724, 0.49780366, 0.49776983]
        std = [0.09050678, 0.09017131, 0.0898702]
        size = config.sample_size

        spatial_transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        # spatial_transform = Compose([crop_method, ToTensor(), norm])
        temporal_transform = SequentialCrop(size=config.sample_duration, stride=config.stride, overlap=config.overlap, max_segments=max_segments)
      
    
    # if pretrained:
    #     if device == torch.device('cpu'):
    #         state_dict = torch.load(pretrained, map_location=device)    
    #     else:
    #         state_dict = torch.load(pretrained)
    #     model.load_state_dict(state_dict, strict=False)
    return model, spatial_transform, temporal_transform

def load_anomaly_detector(config: Config, source):
    model, _ = AnomalyDetector_model(config, source) 
    # model = AnomalyDetector(input_dim=input_dim)
    # if pretrained:
    #     if device == torch.device('cpu'):
    #         checkpoint = torch.load(pretrained, map_location=device)    
    #     else:
    #         checkpoint = torch.load(pretrained)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    return model


def main(config: Config, source, dataset_dir, output_csvpath):
    # load trained model
    print("*** loading model from {model}".format(model = config.pretrained_model))
    feature_extractor, spatial_transform, temporal_transform = load_feature_extractor(config, source)
    # feature_extractor.to(device)
    anomaly_detector = load_anomaly_detector(config, source)
    # anomaly_detector.to(device)
    
    print("*** calculating the model output of the images in {img_dir}"
            .format(img_dir = dataset_dir))
    
    for i, video_name in enumerate(os.listdir(dataset_dir)):
        pt = os.path.join(dataset_dir, video_name)
        if not os.path.isdir(pt):
            continue
        
        out_pth = os.path.join(output_csvpath,video_name+'.csv')
        print(i, pt, "-->",out_pth)
        # df = eval_one_dir(pt, model)
        if not os.path.exists(out_pth):
            df = eval_one_dir_an(config, pt, feature_extractor, spatial_transform, temporal_transform, anomaly_detector) ##Change this to use the classifier or the anomaly_detector
            df.to_csv(out_pth, index = False)
    

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # pretrained_feature_extractor = "/Users/davidchoqueluqueroman/Documents/CODIGOS/AVSS2019/src/MyTrainedModels/resnetXT_fps10_hmdb511_52_0.3863_2.666646_SDI.pth"
    # pretrained_model = "/Users/davidchoqueluqueroman/Documents/CODIGOS/AVSS2019/src/MyTrainedModels/anomaly_detector_datasetUCFCrime2Local_epochs100000-epoch-19000.pth"

    # pretrained_feature_extractor = ("/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/src/MyTrainedModels/resnetXT_fps10_hmdb511_52_0.3863_2.666646_SDI.pth",
    #                                 "/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/src/VioNet/weights/S3D_kinetics400.pt")

    # pretrained_feature_extractor = ("/content/drive/My Drive/VIOLENCE DATA/MyTrainedModels/resnetXT_fps10_hmdb511_52_0.3863_2.666646_SDI.pth",
    #                                 "/content/VioDenseDuplication/src/VioNet/weights/S3D_kinetics400.pt")
    pretrained_feature_extractor = "/content/drive/My Drive/VIOLENCE DATA/MyTrainedModels/resnetXT_fps10_hmdb511_52_0.3863_2.666646_SDI.pth"

    pretrained_model = "/content/drive/My Drive/VIOLENCE DATA/VioNet_pth/anomaly-det_dataset(UCFCrime2LocalClips)_epochs(100000)/anomaly-det_dataset(UCFCrime2LocalClips)_epochs(100000)_resnetxt-epoch-40000.chk"
    # pretrained_model = "/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/VioNet_pth/anomaly-det_dataset(ucfcrime2local)_epochs(100000)_resnetxt+s3d-restore-1-epoch-30000.chk"

    _, anomaly_detec_name = os.path.split(pretrained_model)
    
    config = Config(model="anomaly-det",
                    dataset="hockey",
                    device=device,
                    input_mode='rgb',
                    sample_duration=10,
                    stride=1,
                    overlap=0,
                    sample_size=(224,224),
                    val_batch=4,
                    input_dimension=512,#(512,1024),
                    pretrained_fe=pretrained_feature_extractor,
                    pretrained_model=pretrained_model)
    
    # dataset_dir = "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames/val/Fight"
    # output_csvpath = "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/rwf-vscores/val/Fight"
    source = FEAT_EXT_RESNEXT

    if config.dataset == "rwf-2000":
        splits = ["train/Fight", "train/NonFight", "val/Fight","val/NonFight"]
        folder_out = "/content/drive/My Drive/VIOLENCE DATA/scores-dataset({})-ANmodel({})-input({})".format(config.dataset, anomaly_detec_name[:-4], config.input_mode)
        # folder_out = "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/scores-dataset({})-ANmodel({})-input({})".format(config.dataset, anomaly_detec_name[:-4], config.input_mode)
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)
            for s in splits:
                os.makedirs(os.path.join(folder_out,s))
        for s in splits:
            dataset_dir = "/content/DATASETS/rwf-2000_jpg/frames/{}".format(s)
            # dataset_dir = "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames/{}".format(s)
            output_csvpath="{}/{}".format(folder_out, s)
            main(config, source, dataset_dir, output_csvpath)
    elif config.dataset == "hockey":
        splits = ["fi", "no"]
        folder_out = "/content/drive/My Drive/VIOLENCE DATA/scores-dataset({})-ANmodel({})-input({})".format(config.dataset, anomaly_detec_name[:-4], config.input_mode)
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)
            for s in splits:
                os.makedirs(os.path.join(folder_out,s))
        for s in splits:
            dataset_dir = "/content/DATASETS/hockey_jpg/{}".format(s)
            output_csvpath="{}/{}".format(folder_out, s)
            main(config, source, dataset_dir, output_csvpath)
