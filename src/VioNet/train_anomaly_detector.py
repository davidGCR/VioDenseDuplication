import os
import sys
g_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(g_path)
sys.path.insert(1, g_path)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch

from FeatureExtraction.features_loader import FeaturesLoader, ConcatFeaturesLoader
from VioNet.global_var import getFolder
from VioNet.config import Config
from VioNet.model import AnomalyDetector_model as AN
from VioNet.models.anomaly_detector import custom_objective, RegularizedLoss
from VioNet.epoch import train_regressor
from utils import Log, save_checkpoint, load_checkpoint

def main(config: Config, source, features_path, annotation_path):
    if source == "resnetxt":
        data = FeaturesLoader(features_path=features_path,
                          annotation_path=annotation_path,
                          bucket_size=config.bag_size,
                          features_dim=config.input_dimension)
    elif source == "resnetxt+s3d":
        data = ConcatFeaturesLoader(features_path_1=features_path[0],
                                         features_path_2=features_path[1],
                                         annotation_path=annotation_path,
                                         bucket_size=config.bag_size,
                                         features_dim_1=config.input_dimension[0],
                                         features_dim_2=config.input_dimension[1],
                                         metadata=True)
    

    # feature, label = data[0]
    # print("feature:", feature.size())
    # print("label:", label)
    loader = DataLoader(data,
                        batch_size=config.train_batch,
                        shuffle=True,
                        num_workers=4)
    
    template_log = "/{}_dataset({})_epochs({})_{}".format(config.model,config.dataset,config.num_epoch, config.additional_info)
    log_path = getFolder('VioNet_log')
    chk_path = getFolder('VioNet_pth')

    #create a folder to checkpoints
    chk_path = os.path.join(chk_path, "{}_dataset({})_epochs({})".format(config.model,config.dataset, config.num_epoch))
    if not os.path.isdir(chk_path):
        os.mkdir(chk_path)

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
    model, params = AN(config, source)

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
        model='anomaly-det',
        dataset='ucfcrime2local',
        device=device,
        num_epoch=100000,
        save_every=1000,
        learning_rate=0.01,
        train_batch=60,
        bag_size=30
    )
    source = "resnetxt+s3d"#resnetxt , resnetxt+s3d

    if source == "resnetxt":
        features_path="/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/features_input(dynamic-images)_frames(16)"#"/content/DATASETS/UCFCrime2Local/features_input(dynamic-images)_frames(16)"#"/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/features2D-train"#
        #"/content/DATASETS/UCFCrime2Local/ucfcrime2local_train_ann.txt"#"rwf-2000-train_ann.txt"#
        config.input_dimension=512
    elif source == "resnetxt+s3d":
        features_path = ("/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/features_input(dynamic-images)_frames(16)",
                         "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/features_S3D_input(rgb)_frames(16)")
        config.input_dimension=(512,1024)
    config.additional_info = source
    annotation_path="/Users/davidchoqueluqueroman/Documents/CODIGOS/AVSS2019/ucfcrime2local_train_ann.txt"

    
    ##pretrined model INICIALIZATION
    # config.pretrained_model = "model_final_100000.weights"

    #restore training
    config.restore_training = False
    # config.checkpoint_path = "/Users/davidchoqueluqueroman/Documents/CODIGOS/AVSS2019/VioNet_pth/anomaly_detector_datasetrwf-2000_epochs200000_no-pretrained-model-restore4-epoch-151000.chk"


    main(config, source, features_path, annotation_path)