import os
from models.models2D import ResNet, Densenet2D
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

def eval_one_dir(img_dir, model):
  
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

    outputs = []
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

        # print("output.size(): ",output.size(), output, pred)
        # print("preds:", pred)
        outputs.append(pred.cpu().data.numpy())
        imgpaths += segment

    # print('len(imgpaths):', len(imgpaths))

    df = pd.DataFrame(np.zeros((len(data_loader.dataset), 2)))
    df.columns = ["imgpath", "pred"]
    df['imgpath'] = imgpaths
    # 
    # print('len(outputs):', len(outputs))
   
    # print(np.concatenate(outputs))
    df.iloc[:,1:] = np.concatenate(outputs)
    # df.sort_values(by = 'imgpath', inplace=True)
    # print(df.head())
    return df

def main():

    # load trained model
    print("*** loading model from {model}".format(model = args.model))
    model = Densenet2D(num_classes=2)
    model = model.to(device)
    print('device:', device, type(device))

    if device==torch.device('cpu'):
        state_dict = torch.load(args.model, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(args.model)
    
    model.load_state_dict(state_dict)
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
            df = eval_one_dir(pt, model)
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