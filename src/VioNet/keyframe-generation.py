import os
from models.models2D import ResNet
from dataset import VioDB, ProtestDatasetEval
from torch.utils.data import DataLoader
import tqdm
import pandas as pd
import numpy as np
import torch
import argparse
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sm = torch.nn.Softmax()

def eval_one_dir(img_dir, model):
    """
    return model output of all the images in a directory
    """
    model.eval()
    # make dataloader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = ProtestDatasetEval(
                    img_dir = img_dir,
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]))
    data_loader = DataLoader(dataset,
                            num_workers = 4,
                            batch_size = args.batch_size,
                            shuffle=False)

    outputs = []
    imgpaths = []

    n_imgs = len(os.listdir(img_dir))
    # with tqdm(total=n_imgs) as pbar:
    for i, sample in enumerate(data_loader):
        imgpath, input = sample
        input_var = input.to(device)

        # input_var = Variable(input)
        output = model(input_var)
        probabilities = sm(output) 

        # print("output.size(): ",output.size(), output)
        outputs.append(probabilities.cpu().data.numpy())
        imgpaths += imgpath
        # if i < n_imgs / args.batch_size:
        #     pbar.update(args.batch_size)
        # else:
        #     pbar.update(n_imgs%args.batch_size)


    df = pd.DataFrame(np.zeros((len(os.listdir(img_dir)), 3)))
    # df.columns = ["imgpath", "protest", "violence", "sign", "photo",
    #                 "fire", "police", "children", "group_20", "group_100",
    #                 "flag", "night", "shouting"]
    df.columns = ["imgpath", "no_violence", "violence"]
    df['imgpath'] = imgpaths
    # print('len(imgpaths):', len(imgpaths))
    # print('len(outputs):', len(outputs))
    # print(df.head())
    # print(np.concatenate(outputs))
    df.iloc[:,1:] = np.concatenate(outputs)
    df.sort_values(by = 'imgpath', inplace=True)
    return df

def main():

    # load trained model
    print("*** loading model from {model}".format(model = args.model))
    model = ResNet(num_classes=2)
    model = model.to(device)

    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict)
    print("*** calculating the model output of the images in {img_dir}"
            .format(img_dir = args.dataset_dir))

    # calculate output
    # df = eval_one_dir(args.img_dir, model)
    # write csv file
    # df.to_csv(args.output_csvpath, index = False)
    i=0
    for video_name in os.listdir(args.dataset_dir):
        pt = os.path.join(args.dataset_dir, video_name)
        if not os.path.isdir(pt):
            continue
        print(i, pt)
        
        out_pth = os.path.join(args.output_csvpath,video_name+'.csv')
        if not os.path.exists(out_pth):
            df = eval_one_dir(pt, model)
            df.to_csv(out_pth, index = False)
        i+=1
    

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