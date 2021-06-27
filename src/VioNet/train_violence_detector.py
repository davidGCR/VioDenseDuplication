
from torch._C import device
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
# device = get_torch_device()

def main(config: Config):
    device = config.device
    make_dataset = MakeRWF2000(root='/media/david/datos/Violence DATA/RWF-2000/frames',#'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames', 
                                train=True,
                                path_annotations='/media/david/datos/Violence DATA/Tubes/RWF-2000')#'/Users/davidchoqueluqueroman/Documents/DATASETS_Local/Tubes/RWF-2000')
    dataset = TubeDataset(frames_per_tube=16, 
                            min_frames_per_tube=8, 
                            make_function=make_dataset,
                            spatial_transform=transforms.Compose([
                                transforms.CenterCrop(224),
                                transforms.ToTensor()
                            ]),
                            max_num_tubes=4)
    loader = DataLoader(dataset,
                        batch_size=4,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True,
                        collate_fn=my_collate)

    model, params = ViolenceDetector_model(config, device)
    # print(model)
    # optimizer = torch.optim.Adadelta(params, lr=config.learning_rate, eps=1e-8)

    # criterion = RegularizedLoss(model, custom_objective)
    
    ##iterate over dataset
    for i, data in enumerate(loader):
        # path, label, annotation,frames_names, boxes, video_images = data
        boxes, video_images, labels = data
        # boxes, video_images, labels = boxes.to(device), video_images.to(device), labels.to(device)
        print('_____ {} ______'.format(i+1))
        
        print('boxes: ', type(boxes), len(boxes), '-boxes[0]: ', boxes[0].size())
        print('video_images: ', type(video_images), len(video_images), '-video_images[0]: ', video_images[0].size())
        print('labels: ', type(labels), len(labels), '-labels: ', labels)

        video_images = [torch.rand(v.size()[0],3,16,224,224) for v in video_images]
        scores = []
        for i in range(1):
            video_images[i] = video_images[i].to(device)
            boxes[i] = boxes[i].to(device)
            # labels[i] = labels[i].to(device)

            print('boxes[{}]: '.format(i), boxes[i], boxes[i].size())
            print('video_images[{}]: '.format(i), video_images[i].size())
            print('labels[{}]: '.format(i), labels[i])

            y_pred = model(video_images[i], boxes[i])
            scores.append(y_pred)
        print('Scores: ', scores)
        if i==1:
            break

if __name__=='__main__':
    config = Config(
        model='',
        dataset='',
        device=get_torch_device()
    )
    main(config)