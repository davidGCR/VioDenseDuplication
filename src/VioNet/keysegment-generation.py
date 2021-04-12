import os
from models.models2D import ResNet, Densenet2D, FeatureExtractorResNet, FeatureExtractorResNextTempPool
from models.anomaly_detector import AnomalyDetector
from dataset import VioDB, ProtestDatasetEval, RwfDatasetEval
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
from config import Config

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

def eval_one_dir_an(config: Config, img_dir, feature_extractor, anomaly_detector):
  
    """
    return model output of all the images in a directory
    """
    
    feature_extractor.eval()
    anomaly_detector.eval()
    # make dataloader
    sample_size = config.sample_size
    crop_method = GroupScaleCenterCrop(size=sample_size)
    norm = Normalize([0.49778724, 0.49780366, 0.49776983], [0.09050678, 0.09017131, 0.0898702 ])
    dataset = config.dataset
    # cv = 1
    sample_duration = config.sample_duration
    stride = config.stride
    # input_mode = "dynamic-images"
    overlap = config.overlap
    batch_size = config.val_batch

    spatial_transform = Compose([crop_method, ToTensor(), norm])
    target_transform = Label()
    temporal_transform = SequentialCrop(size=sample_duration, stride=stride, overlap=overlap)

    val_dataset = RwfDatasetEval(img_dir, spatial_transform, temporal_transform)

    data_loader = DataLoader(val_dataset,
                            num_workers = 4,
                            batch_size = batch_size,
                            shuffle=False)

    predictions = []
    imgpaths = []
    for i, sample in enumerate(data_loader):
        input, segment = sample

        input = torch.squeeze(input, 2)
        # print('segment:', segment)
        # print('input:', input.size())
        input_var = input.to(device)
        feature = feature_extractor(input_var)
        score = anomaly_detector(feature)

        # print("feature: ", feature.size())
        # print("score: ", score.size())
        # _, pred = output.topk(1, 1, True)
        # probabilities = sm(output) 
        result = score.cpu().data.numpy()
        predictions.append(result)
        imgpaths += segment

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

def load_feature_extractor(model_type="resnet", pretrained=None):
    if model_type == "resnet":
        model = FeatureExtractorResNet()
    elif model_type == "resnetXT":
        model = FeatureExtractorResNextTempPool()
    
    if pretrained:
        if device == torch.device('cpu'):
            state_dict = torch.load(pretrained, map_location=device)    
        else:
            state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict, strict=False)
    return model

def load_anomaly_detector(input_dim, pretrained):
    
    model = AnomalyDetector(input_dim=input_dim)
    if pretrained:
        if device == torch.device('cpu'):
            checkpoint = torch.load(pretrained, map_location=device)    
        else:
            checkpoint = torch.load(pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model


def main(config: Config, dataset_dir, output_csvpath):
    # load trained model
    print("*** loading model from {model}".format(model = config.pretrained_model))
    feature_extractor = load_feature_extractor(config.model, config.pretrained_fe)
    feature_extractor.to(device)
    anomaly_detector = load_anomaly_detector(config.input_dimension, config.pretrained_model)
    anomaly_detector.to(device)
    
    print("*** calculating the model output of the images in {img_dir}"
            .format(img_dir = dataset_dir))
    
    for i, video_name in enumerate(os.listdir(dataset_dir)):
        pt = os.path.join(dataset_dir, video_name)
        if not os.path.isdir(pt):
            continue
        print(i, pt)
        
        out_pth = os.path.join(output_csvpath,video_name+'.csv')
        # df = eval_one_dir(pt, model)
        if not os.path.exists(out_pth):
            df = eval_one_dir_an(config, pt, feature_extractor, anomaly_detector) ##Change this to use the classifier or the anomaly_detector
            df.to_csv(out_pth, index = False)
    

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # pretrained_feature_extractor = "/Users/davidchoqueluqueroman/Documents/CODIGOS/AVSS2019/src/MyTrainedModels/resnetXT_fps10_hmdb511_52_0.3863_2.666646_SDI.pth"
    # pretrained_model = "/Users/davidchoqueluqueroman/Documents/CODIGOS/AVSS2019/src/MyTrainedModels/anomaly_detector_datasetUCFCrime2Local_epochs100000-epoch-19000.pth"

    pretrained_feature_extractor = "/content/drive/My Drive/VIOLENCE DATA/VioNet_pth/resnetXT_fps10_hmdb511_52_0.3863_2.666646_SDI.pth"
    pretrained_model = "/content/drive/My Drive/VIOLENCE DATA/VioNet_pth/anomaly_detector_datasetUCFCrime2Local_epochs100000-epoch-19000.pth"

    _, anomaly_detec_name = os.path.split(pretrained_model)
    
    config = Config(model="resnetXT",
                    dataset="rwf-2000",
                    device=device,
                    input_mode='rgb',
                    sample_duration=16,
                    stride=1,
                    overlap=0,
                    sample_size=(224,224),
                    val_batch=16,
                    input_dimension=512,
                    pretrained_fe=pretrained_feature_extractor,
                    pretrained_model=pretrained_model)
    
    # dataset_dir = "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames/val/Fight"
    # output_csvpath = "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/rwf-vscores/val/Fight"
    
    splits = ["train/Fight", "train/NonFight", "val/Fight","val/NonFight"]
    folder_out = "/content/drive/My Drive/VIOLENCE DATA/scores-dataset({})-ANmodel({})-input({})".format(config.dataset, anomaly_detec_name[:-4], config.input_mode)
    if not os.path.isdir(folder_out):
        os.mkdir(folder_out)
        for s in splits:
            os.makedirs(os.path.join(folder_out,s))
    for s in splits:
        dataset_dir = "/content/DATASETS/{}".format(s)
        output_csvpath="{}/{}".format(folder_out, s)
        main(config, dataset_dir, output_csvpath)
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("--img_dir",
    # #                     type=str,
    # #                     help = "image directory to calculate output"
    # #                     "(the directory must contain only image files)",
    # #                     default=''
    # #                     )

    # parser.add_argument("--dataset_dir",
    #                     type=str,
    #                     required = True,
    #                     help = "dataset directory to calculate output"
    #                     "(the directory must contain only image files)"
    #                     )

    # parser.add_argument("--output_csvpath",
    #                     type=str,
    #                     default = "result.csv",
    #                     help = "path to output csv file"
    #                     )
    # parser.add_argument("--model",
    #                     type=str,
    #                     required = True,
    #                     help = "model path"
    #                     )
    # # parser.add_argument("--cuda",
    # #                     action = "store_true",
    # #                     help = "use cuda?",
    # #                     default=None
    # #                     )
    # parser.add_argument("--workers",
    #                     type = int,
    #                     default = 4,
    #                     help = "number of workers",
    #                     )
    # parser.add_argument("--batch_size",
    #                     type = int,
    #                     default = 16,
    #                     help = "batch size",
    #                     )
    # args = parser.parse_args()

    # main()