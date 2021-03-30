import os
import sys
g_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(g_path)
sys.path.insert(1, g_path)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch

from FeatureExtraction.features_loader import FeaturesLoader
from VioNet.global_var import getFolder
from VioNet.config import Config
from VioNet.model import AnomalyDetector_model as AN
from VioNet.models.anomaly_detector import custom_objective, RegularizedLoss
from VioNet.epoch import train_regressor
from utils import Log

def main(config: Config):
    data = FeaturesLoader(features_path="/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/features2D",
                          annotation_path="/Users/davidchoqueluqueroman/Documents/CODIGOS/AVSS2019/test_ann.txt",
                          bucket_size=config.bag_size,
                          features_dim=config.input_dimension)

    # feature, label = data[0]
    # print("feature:", feature.size())
    # print("label:", label)
    loader = DataLoader(data,
                        batch_size=config.train_batch,
                        shuffle=False,
                        num_workers=4)
    
    template_log = "/anomaly_detector_dataset{}_epochs{}".format(config.dataset,config.num_epoch)
    log_path = getFolder('VioNet_log')
    chk_path = getFolder('VioNet_pth')
    tsb_path = getFolder('VioNet_tensorboard_log')
    log_tsb_dir = tsb_path + template_log
    for pth in [log_path, chk_path, tsb_path, log_tsb_dir]:
        # make dir
        if not os.path.exists(pth):
            os.mkdir(pth)

    print('tensorboard dir:', log_tsb_dir)                                                
    writer = SummaryWriter(log_tsb_dir)

    # log
    # batch_log = Log(log_path+template_log +".csv".format(config.dataset,config.num_epoch),['epoch', 'batch', 'iter', 'loss', 'lr'])
    epoch_log = Log(log_path+template_log +".csv",['epoch', 'loss', 'lr'])
    
    ## train parameters and others
    # criterion = nn.CrossEntropyLoss().to(device)
    # learning_rate = config.learning_rate
    # momentum = config.momentum
    # weight_decay = config.weight_decay
    # optimizer = torch.optim.SGD(params=params,
    #                             lr=learning_rate,
    #                             momentum=momentum,
    #                             weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                        verbose=True,
    #                                                        factor=config.factor,
    #                                                        min_lr=config.min_lr)

    #model
    model, params = AN(config)

    # Training parameters
    """
    In the original paper:
        lr = 0.01
        epsilon = 1e-8
    """
    optimizer = torch.optim.Adadelta(params, lr=config.learning_rate, eps=1e-8)

    criterion = RegularizedLoss(model, custom_objective)
    loss_baseline = 1

    for i in range(config.num_epoch):
        epoch=i+1
        train_loss, lr = train_regressor(epoch=i, 
                                        data_loader=loader, 
                                        model=model, 
                                        criterion=criterion, 
                                        optimizer=optimizer, 
                                        device=config.device,
                                        epoch_log=epoch_log)
        writer.add_scalar('training loss',
                            train_loss,
                            epoch)
        if epoch%config.save_every == 0:
            torch.save(model.state_dict(), chk_path+template_log+"-epoch-"+str(epoch)+".pth")

    


if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = Config(
        model='resnetXT',  # c3d, convlstm, densenet, densenet_lean, resnet50, densenet2D, resnetXT
        dataset='UCFCrime2Local',
        device=device,
        num_epoch=100,
        save_every=25,
        learning_rate=0.01,
        input_dimension=512,
        train_batch=60,
        bag_size=30
    )

    main(config)