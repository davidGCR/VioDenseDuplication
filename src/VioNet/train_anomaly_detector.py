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
from utils import Log, save_checkpoint, load_checkpoint

def main(config: Config, features_path, annotation_path):
    data = FeaturesLoader(features_path=features_path,
                          annotation_path=annotation_path,
                          bucket_size=config.bag_size,
                          features_dim=config.input_dimension)

    # feature, label = data[0]
    # print("feature:", feature.size())
    # print("label:", label)
    loader = DataLoader(data,
                        batch_size=config.train_batch,
                        shuffle=True,
                        num_workers=4)
    
    template_log = "/anomaly_detector_dataset{}_epochs{}_{}".format(config.dataset,config.num_epoch, config.additional_info)
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
    epoch_log = Log(log_path+template_log +".csv",['epoch', 'loss', 'lr'])
    
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

    last_epoch = 0
    ##Restore training
    if config.restore_training:
        model, optimizer, last_epoch, last_loss = load_checkpoint(model, config.device, optimizer, config.checkpoint_path)
    

    for i in range(config.num_epoch-last_epoch):
        epoch=last_epoch+i+1
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
            save_checkpoint(model, epoch, optimizer, train_loss, chk_path+template_log+"-epoch-"+str(epoch)+".chk")
            # torch.save(model.state_dict(), chk_path+template_log+"-epoch-"+str(epoch)+".pth")

    


if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = Config(
        model='resnetXT',  # c3d, convlstm, densenet, densenet_lean, resnet50, densenet2D, resnetXT
        dataset='UCFCrime2Local',
        device=device,
        num_epoch=10000,
        save_every=500,
        learning_rate=0.01,
        input_dimension=512,
        train_batch=60,
        bag_size=30
    )
    # features_path="/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/features2D",
    # annotation_path="/Users/davidchoqueluqueroman/Documents/CODIGOS/AVSS2019/test_ann.txt",
    features_path="/content/DATASETS/UCFCrime2Local/features2D"
    annotation_path="/content/DATASETS/UCFCrime2Local/test_ann.txt"
    config.additional_info = ""

    ##pretrined model INICIALIZATION
    config.pretrained_model = "model_final_100000.weights"

    #restore training
    config.restore_training = False
    config.checkpoint_path = ""


    main(config, features_path, annotation_path)