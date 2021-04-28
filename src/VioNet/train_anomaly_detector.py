import os
import sys
g_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(g_path)
sys.path.insert(1, g_path)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch

from FeatureExtraction.features_loader import FeaturesLoader, ConcatFeaturesLoader
from VioNet.config import Config
from VioNet.model import AnomalyDetector_model as AN
from VioNet.models.anomaly_detector import custom_objective, RegularizedLoss
from VioNet.epoch import train_regressor
from utils import Log, save_checkpoint, load_checkpoint

def main(config: Config, source, features_path, annotation_path, home_path):
    if source == FEAT_EXT_RESNEXT:
        data = FeaturesLoader(features_path=features_path,
                          annotation_path=annotation_path,
                          bucket_size=config.bag_size,
                          features_dim=config.input_dimension)
    elif source == FEAT_EXT_RESNEXT_S3D:
        data = ConcatFeaturesLoader(features_path_1=features_path[0],
                                         features_path_2=features_path[1],
                                         annotation_path=annotation_path,
                                         bucket_size=config.bag_size,
                                         features_dim_1=config.input_dimension[0],
                                         features_dim_2=config.input_dimension[1],
                                         metadata=True)
    elif source == FEAT_EXT_C3D:
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
    
    template_log = "{}_Dataset({})_Features({})_TotalEpochs({})_ExtraInfo({})".format(config.model,
                                                                                        config.dataset,
                                                                                        os.path.split(features_path)[1],
                                                                                        config.num_epoch,
                                                                                        config.additional_info)
    log_path = os.path.join(home_path, PATH_LOG, template_log)
    chk_path = os.path.join(home_path, PATH_CHECKPOINT, template_log)
    tsb_path = os.path.join(home_path, PATH_TENSORBOARD, template_log)

    for pth in [log_path, chk_path, tsb_path]:
        # make dir
        if not os.path.exists(pth):
            os.mkdir(pth)

    print('tensorboard dir:', tsb_path)                                                
    writer = SummaryWriter(tsb_path)

    # log
    epoch_log = Log(os.path.join(log_path, template_log +"-LOG.csv"),['epoch', 'loss', 'lr'])
    
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

    
from utils import get_torch_device
from global_var import *

if __name__=="__main__":
    device = get_torch_device()
    config = Config(
        model=MODEL_ANOMALY_DET,
        dataset=UCFCrime2LocalClips_DATASET,
        device=device,
        num_epoch=100000,
        save_every=2000,
        learning_rate=0.01,
        train_batch=64,
        bag_size=32
    )
    source = FEAT_EXT_C3D#resnetxt , resnetxt+s3d

    enviroment_config = {
        "home": HOME_UBUNTU
    }

    if source == FEAT_EXT_RESNEXT:
        # features_path="/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/features_input(dynamic-images)_frames(16)"#"/content/DATASETS/UCFCrime2Local/features_input(dynamic-images)_frames(16)"#"/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/features2D-train"#
        features_path="/content/DATASETS/UCFCrime2Local/features_from(ucfcrime2localClips)_input(dynamic-images)_frames(10)_num_segments(32)"
        config.input_dimension=512
    elif source == FEAT_EXT_RESNEXT_S3D:
        # features_path = ("/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/features_input(dynamic-images)_frames(16)",
        #                  "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/features_S3D_input(rgb)_frames(16)")
        features_path = ("/content/DATASETS/UCFCrime2Local/features_input(dynamic-images)_frames(16)",
                         "/content/DATASETS/UCFCrime2Local/features_S3D_input(rgb)_frames(16)")
        config.input_dimension=(512,1024)
    elif source == FEAT_EXT_C3D:
        features_path = os.path.join(enviroment_config["home"], "ExtractedFeatures", "Features_Dataset(UCFCrime2LocalClips)_FE(c3d)_Input(rgb)_Frames(16)_Num_Segments(32)")
        config.input_dimension = 4096
    config.additional_info = source
    annotation_path = os.path.join(enviroment_config["home"], config.dataset, "UCFCrime2LocalClips-train_ann.txt")
    # annotation_path="/Users/davidchoqueluqueroman/Documents/CODIGOS/AVSS2019/ucfcrime2local_train_ann.txt"
    
    ##pretrined model INICIALIZATION
    # config.pretrained_model = "model_final_100000.weights"

    #restore training
    config.restore_training = False
    # config.checkpoint_path = "/content/drive/MyDrive/VIOLENCE DATA/VioNet_pth/anomaly-det_dataset(UCFCrime2LocalClips)_epochs(100000)/anomaly-det_dataset(UCFCrime2LocalClips)_epochs(100000)_resnetxt-epoch-49000.chk"

    main(config, source, features_path, annotation_path, enviroment_config["home"])