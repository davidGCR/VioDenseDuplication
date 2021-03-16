import os
from models.models2D import ResNet, Densenet2D, FeatureExtractorResNet
from models.anomaly_detector import AnomalyDetector
from dataset import VioDB, ProtestDatasetEval, RwfDatasetEval
from torch.utils.data import DataLoader
import tqdm
import pandas as pd
import numpy as np
import torch
import argparse
import torchvision.transforms as transforms
from spatial_transforms import Compose, ToTensor, Normalize, GroupScaleCenterCrop
from temporal_transforms import CenterCrop, SequentialCrop
from target_transforms import Label

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

def eval_one_dir_an(img_dir, feature_extractor, anomaly_detector):
  
    """
    return model output of all the images in a directory
    """
    
    feature_extractor.eval()
    anomaly_detector.eval()
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
    batch_size = args.batch_size

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

def load_feature_extractor(model_type="resnet"):
    if model_type == "resnet":
        model = FeatureExtractorResNet()
    return model

def load_anomaly_detector(pretrained=None):
    
    model = AnomalyDetector(input_dim=2048)
    if pretrained:
        state_dict = torch.load(pretrained, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    return model

def main():

    # load trained model
    print("*** loading model from {model}".format(model = args.model))
    feature_extractor = load_feature_extractor()
    feature_extractor.to(device)
    anomaly_detector = load_anomaly_detector(pretrained=args.model)
    anomaly_detector.to(device)
    # print('device:', device, type(device))

    
    print("*** calculating the model output of the images in {img_dir}"
            .format(img_dir = args.dataset_dir))

    
    for i, video_name in enumerate(os.listdir(args.dataset_dir)):
        pt = os.path.join(args.dataset_dir, video_name)
        if not os.path.isdir(pt):
            continue
        print(i, pt)
        
        out_pth = os.path.join(args.output_csvpath,video_name+'.csv')
        # df = eval_one_dir(pt, model)
        if not os.path.exists(out_pth):
            df = eval_one_dir_an(pt, feature_extractor, anomaly_detector) ##Change this to use the classifier or the anomaly_detector
            df.to_csv(out_pth, index = False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--img_dir",
    #                     type=str,
    #                     help = "image directory to calculate output"
    #                     "(the directory must contain only image files)",
    #                     default=''
    #                     )

    parser.add_argument("--dataset_dir",
                        type=str,
                        required = True,
                        help = "dataset directory to calculate output"
                        "(the directory must contain only image files)"
                        )

    parser.add_argument("--output_csvpath",
                        type=str,
                        default = "result.csv",
                        help = "path to output csv file"
                        )
    parser.add_argument("--model",
                        type=str,
                        required = True,
                        help = "model path"
                        )
    # parser.add_argument("--cuda",
    #                     action = "store_true",
    #                     help = "use cuda?",
    #                     default=None
    #                     )
    parser.add_argument("--workers",
                        type = int,
                        default = 4,
                        help = "number of workers",
                        )
    parser.add_argument("--batch_size",
                        type = int,
                        default = 16,
                        help = "batch size",
                        )
    args = parser.parse_args()

    main()