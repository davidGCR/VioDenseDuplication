
from torch.utils.tensorboard import SummaryWriter
import torch

#data
from customdatasets.make_dataset import MakeRWF2000
from customdatasets.tube_dataset import TubeDataset, my_collate
from torchvision import transforms
from torch.utils.data import DataLoader, dataset

from config import Config
from model import ViolenceDetector_model
from models.anomaly_detector import custom_objective, RegularizedLoss
from epoch import train_regressor
from utils import Log, save_checkpoint, load_checkpoint
from utils import get_torch_device
device = get_torch_device()

def main(config: Config):
    make_dataset = MakeRWF2000(root='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames', 
                                train=True,
                                path_annotations='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/Tubes/RWF-2000')
    dataset = TubeDataset(frames_per_tube=16, 
                            min_frames_per_tube=8, 
                            make_function=make_dataset,
                            spatial_transform=transforms.Compose([
                                transforms.CenterCrop(224),
                                transforms.ToTensor()
                            ]))
    loader = DataLoader(dataset,
                        batch_size=4,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True,
                        collate_fn=my_collate)

    model, params = ViolenceDetector_model(config, device)
    print(model)
    
    ##iterate over dataset
    for i, data in enumerate(loader):
        # path, label, annotation,frames_names, boxes, video_images = data
        boxes, video_images, labels = data
        print('_____ {} ______'.format(i+1))
        # print('path: ', path)
        # print('label: ', label)
        # print('annotation: ', annotation)
        print('boxes: ', type(boxes), len(boxes), '-boxes[0]: ', boxes[0].size())
        print('video_images: ', type(video_images), len(video_images), '-video_images[0]: ', video_images[0].size())
        print('labels: ', type(labels), len(labels), '-labels: ', labels)

if __name__=='__main__':
    config = Config(
        model='',
        dataset='',
        device=''
    )
    main(config)