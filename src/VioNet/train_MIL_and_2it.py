# def load_features(config: Config):
#     device = config.device
#     make_dataset = MakeRWF2000(root='/media/david/datos/Violence DATA/RWF-2000/frames',
#                                 train=False,
#                                 path_annotations='/media/david/datos/Violence DATA/Tubes/RWF-2000',
#                                 path_feat_annotations='/media/david/datos/Violence DATA/i3d-FeatureMaps/rwf'
#                                 )
#     dataset = TubeFeaturesDataset(frames_per_tube=16,
#                                     min_frames_per_tube=8,
#                                     make_function=make_dataset,
#                                     map_shape=(1,528,4,14,14),
#                                     max_num_tubes=4)
#     loader = DataLoader(dataset,
#                         batch_size=1,
#                         shuffle=False,
#                         num_workers=4,
#                         # pin_memory=True,
#                         )
#     for i, data in enumerate(loader):
#         boxes, f_maps, label = data
#         print('boxes: ', boxes.size())
#         print('f_maps: ', f_maps.size())
#         print('label: ', label.size())

# def extract_features(confi: Config, output_folder: str):
#     device = config.device
#     make_dataset = MakeRWF2000(root='/media/david/datos/Violence DATA/RWF-2000/frames',
#                                 train=False,
#                                 path_annotations='/media/david/datos/Violence DATA/Tubes/RWF-2000')
#     paths, labels, annotations, _ = make_dataset()
#     from models.i3d import InceptionI3d
#     model = InceptionI3d(2, in_channels=3, final_endpoint='Mixed_4e').to(device)
#     load_model_path = '/media/david/datos/Violence DATA/VioNet_weights/pytorch_i3d/rgb_imagenet.pt'
#     state_dict = torch.load(load_model_path)
#     model.load_state_dict(state_dict,  strict=False)
#     model.eval()

#     from models.roi_extractor_3d import SingleRoIExtractor3D
#     roi_op = SingleRoIExtractor3D(roi_layer_type='RoIAlign',
#                                 featmap_stride=16,
#                                 output_size=8,    
#                                 with_temporal_pool=True).to(device)
    
#     features_writer = FeaturesWriter(num_videos=len(paths), num_segments=0)

#     with torch.no_grad():
#         for i in range(len(paths)):
#             video_path = paths[i]
#             annotation_path = annotations[i]
#             print("==={}/{}".format(i+1, video_path))
#             dataset = OneVideoTubeDataset(frames_per_tube=16, 
#                                             min_frames_per_tube=8, 
#                                             video_path=video_path,
#                                             annotation_path=annotation_path,
#                                             spatial_transform=transforms.Compose([
#                                                 # transforms.CenterCrop(224),
#                                                 # transforms.Resize(256),
#                                                 transforms.ToTensor()
#                                             ]),
#                                             max_num_tubes=0)
#             loader = DataLoader(dataset,
#                                 batch_size=1,
#                                 shuffle=False,
#                                 num_workers=4,
#                                 # pin_memory=True,
#                                 # collate_fn=my_collate
#                                 )
#             print('loader len:', len(loader))
#             if len(loader)==0:
#                 continue
#             tmp_names = video_path.split('/')
#             output_file = os.path.join(output_folder, tmp_names[-3], tmp_names[-2], tmp_names[-1])# split, class, video

#             for j, data in enumerate(loader):
#                 box, tube_images = data
#                 tube_images = tube_images.permute(0,2,1,3,4)
#                 box = torch.squeeze(box, dim=1)
                
#                 if box ==None:
#                     print("Noneeee")
#                     continue
                
#                 box = box.to(device)
#                 tube_images = tube_images.to(device)
#                 # print('box: ', box.size())
#                 print('tube_images: ', tube_images.size(), tube_images.device)

#                 f_map = model(tube_images)
#                 print('f_map: ', f_map.size())
#                 features_writer.write(feature=torch.flatten(f_map).cpu().numpy(),
#                                         video_name=tmp_names[-1],
#                                         idx=j,
#                                         dir=os.path.join(output_folder, tmp_names[-3], tmp_names[-2]))
#             features_writer.dump()
            
# def main_2(config: Config):
#     device = config.device
#     make_dataset_nonviolence = load_make_dataset(
#         config.dataset, 
#         train=True,
#         cv_split=config.num_cv, 
#         home_path=config.home_path,
#         category=0,
#         shuffle=True
#         )
#     make_dataset_violence = load_make_dataset(
#         config.dataset, 
#         train=True,
#         cv_split=config.num_cv, 
#         home_path=config.home_path,
#         category=1,
#         shuffle=True
#         )

#     dataset_train_nonviolence = TubeDataset(frames_per_tube=config.frames_per_tube, 
#                             min_frames_per_tube=8,
#                             make_function=make_dataset_nonviolence,
#                             spatial_transform=i3d_transf()['train'],
#                             max_num_tubes=config.num_tubes,
#                             train=True,
#                             dataset=config.dataset,
#                             input_type=config.input_type,
#                             random=config.tube_sampling_random,
#                             keyframe=True,
#                             spatial_transform_2=resnet_transf()['train'])
#     dataset_train_violence = TubeDataset(frames_per_tube=config.frames_per_tube, 
#                             min_frames_per_tube=8,
#                             make_function=make_dataset_violence,
#                             spatial_transform=i3d_transf()['train'],
#                             max_num_tubes=config.num_tubes,
#                             train=True,
#                             dataset=config.dataset,
#                             input_type=config.input_type,
#                             random=config.tube_sampling_random,
#                             keyframe=True,
#                             spatial_transform_2=resnet_transf()['train'])
#     loader_train_nonviolence = DataLoader(dataset_train_nonviolence,
#                         batch_size=config.train_batch,
#                         shuffle=True,
#                         num_workers=config.num_workers,
#                         # pin_memory=True,
#                         collate_fn=my_collate
#                         )
#     loader_train_violence = DataLoader(dataset_train_violence,
#                         batch_size=config.train_batch,
#                         shuffle=True,
#                         num_workers=config.num_workers,
#                         # pin_memory=True,
#                         collate_fn=my_collate
#                         )

#     #validation
#     val_make_dataset_nonviolence = load_make_dataset(
#         config.dataset, 
#         train=False,
#         cv_split=config.num_cv, 
#         home_path=config.home_path,
#         category=0,
#         shuffle=True
#         )
#     val_make_dataset_violence = load_make_dataset(
#         config.dataset, 
#         train=False,
#         cv_split=config.num_cv, 
#         home_path=config.home_path,
#         category=1,
#         shuffle=True
#         )
#     dataset_val_nonviolence = TubeDataset(frames_per_tube=config.frames_per_tube, 
#                             min_frames_per_tube=8,
#                             make_function=val_make_dataset_nonviolence,
#                             spatial_transform=i3d_transf()['val'],
#                             max_num_tubes=config.num_tubes,
#                             train=False,
#                             dataset=config.dataset,
#                             input_type=config.input_type,
#                             random=config.tube_sampling_random,
#                             keyframe=True,
#                             spatial_transform_2=resnet_transf()['val'])
#     dataset_val_violence = TubeDataset(frames_per_tube=config.frames_per_tube, 
#                             min_frames_per_tube=8,
#                             make_function=val_make_dataset_violence,
#                             spatial_transform=i3d_transf()['val'],
#                             max_num_tubes=config.num_tubes,
#                             train=False,
#                             dataset=config.dataset,
#                             input_type=config.input_type,
#                             random=config.tube_sampling_random,
#                             keyframe=True,
#                             spatial_transform_2=resnet_transf()['val'])
#     loader_val_nonviolence = DataLoader(dataset_val_nonviolence,
#                         batch_size=config.train_batch,
#                         shuffle=True,
#                         num_workers=config.num_workers,
#                         # pin_memory=True,
#                         collate_fn=my_collate
#                         )
#     loader_val_violence = DataLoader(dataset_val_violence,
#                         batch_size=config.train_batch,
#                         shuffle=True,
#                         num_workers=config.num_workers,
#                         # pin_memory=True,
#                         collate_fn=my_collate
#                         )
   
#     ################## Full Detector ########################
    
#     #
#     from models.violence_detector import ViolenceDetectorBinary
#     if config.model == 'densenet_lean_roi':
#         model, params = VioNet_densenet_lean_roi(config, config.pretrained_model)
#     elif config.model == 'i3d+roi+i3d':
#         model, params = VioNet_I3D_Roi(config, device, config.pretrained_model)
#     elif config.model == 'i3d+roi+binary':
#         model = ViolenceDetectorBinary(
#             freeze=config.freeze).to(device)
#         params = model.parameters()
#     elif config.model == 'TwoStreamVD_Binary':
#         model = TwoStreamVD_Binary().to(device)
#         params = model.parameters()
#     elif config.model == 'TwoStreamVD_Binary_CFam':
#         model = TwoStreamVD_Binary_CFam(config.model_config).to(device)
#         params = model.parameters()

#     exp_config_log = config.log
    
#     h_p = HOME_DRIVE if config.home_path==HOME_COLAB else config.home_path
#     tsb_path_folder = os.path.join(h_p, PATH_TENSORBOARD, exp_config_log)
#     chk_path_folder = os.path.join(h_p, PATH_CHECKPOINT, exp_config_log)

#     for p in [tsb_path_folder, chk_path_folder]:
#         if not os.path.exists(p):
#             os.makedirs(p)
#     # print('tensorboard dir:', tsb_path)                                                
#     writer = SummaryWriter(tsb_path_folder)

#     if config.optimizer == 'Adadelta':
#         optimizer = torch.optim.Adadelta(params, lr=config.learning_rate, eps=1e-8)
#     elif config.optimizer == 'Adam':
#         optimizer = torch.optim.Adam(params, lr=config.learning_rate)
#     elif config.optimizer == 'SGD':
#         optimizer = torch.optim.SGD(params=params,
#                                     lr=config.learning_rate,
#                                     momentum=0.5,
#                                     weight_decay=1e-3)
    
#     if config.head == REGRESSION:
#         # criterion = nn.BCELoss().to(device)
#         # criterion = nn.BCEWithLogitsLoss().to(device)
#         criterion = MIL
#     elif config.head == BINARY:
#         criterion = nn.CrossEntropyLoss().to(config.device)

#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                            verbose=True,
#                                                            factor=config.factor,
#                                                            min_lr=config.min_lr)
#     from utils import AverageMeter
#     # from epoch import calculate_accuracy_2

#     start_epoch = 0
#     ##Restore training
#     if config.restore_training:
#         model, optimizer, epochs, last_epoch, last_loss = load_checkpoint(model, config.device, optimizer, config.checkpoint_path)
#         start_epoch = last_epoch+1
#         # config.num_epoch = epochs 
    
#     for epoch in range(start_epoch, config.num_epoch):
#         # epoch = last_epoch+i
#         train_loss, train_acc = train_2it(
#             loader_train_violence,
#             loader_train_nonviolence,
#             epoch,
#             model,
#             criterion,
#             optimizer,
#             config.device,
#             config.num_tubes,
#             calculate_accuracy_2,
#             )
#         writer.add_scalar('training loss', train_loss, epoch)
#         writer.add_scalar('training accuracy', train_acc, epoch)
        
#         val_loss, val_acc = val_2it(
#             loader_val_violence,
#             loader_val_nonviolence,
#             epoch,
#             model,
#             criterion,
#             config.device,
#             config.num_tubes,
#             calculate_accuracy_2)
#         scheduler.step(val_loss)
#         writer.add_scalar('validation loss', val_loss, epoch)
#         writer.add_scalar('validation accuracy', val_acc, epoch)

#         if (epoch+1)%config.save_every == 0:
#             save_checkpoint(model, config.num_epoch, epoch, optimizer,train_loss, os.path.join(chk_path_folder,"save_at_epoch-"+str(epoch)+".chk"))

from VioNet.lib.train_script import train_regressor, val_regressor
from config import Config
from global_var import *
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from utils import save_checkpoint, load_checkpoint

 # dataiter = iter(train_loader)
    # video_images, label, path, key_frame, raw_clip_images = dataiter.next()
    # print('raw_clip_images: ', type(raw_clip_images), raw_clip_images.size())
    # # create grid of images
    # img_grid = torchvision.utils.make_grid(raw_clip_images)
    # # show images
    # matplotlib_imshow(img_grid, one_channel=True)
    # plt.show()

    # write to tensorboard
    # writer.add_image('four_fashion_mnist_images', img_grid)

def MIL_training(config: Config, model, dataloader, val_make_dataset, transformations):
    device = config.device
    exp_config_log = config.log
    h_p = HOME_DRIVE if config.home_path==HOME_COLAB else config.home_path
    tsb_path_folder = os.path.join(h_p, PATH_TENSORBOARD, exp_config_log)
    chk_path_folder = os.path.join(h_p, PATH_CHECKPOINT, exp_config_log)

    for p in [tsb_path_folder, chk_path_folder]:
        if not os.path.exists(p):
            os.makedirs(p)                                               
    writer = SummaryWriter(tsb_path_folder)

    if config.optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=config.learning_rate, eps=1e-8)
           
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=config.learning_rate,
                                    momentum=0.5,
                                    weight_decay=1e-3)
    elif config.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr= config.learning_rate, weight_decay=0.0010000000474974513)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
    elif config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr= config.learning_rate)
    
    
    criterion = nn.BCELoss().to(device)

    start_epoch = 0
    ##Restore training
    if config.restore_training:
        model, optimizer, epochs, last_epoch, last_loss = load_checkpoint(model, config.device, optimizer, config.checkpoint_path)
        start_epoch = last_epoch+1
        # config.num_epoch = epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           verbose=True,
                                                           factor=config.factor,
                                                           min_lr=config.min_lr)

    for epoch in range(start_epoch, config.num_epoch):
        train_loss, train_acc = train_regressor(
            dataloader, 
            epoch, 
            model, 
            criterion, 
            optimizer, 
            config.device, 
            config, 
            None,
            False)
        
        val_regressor(val_make_dataset, transformations, model, config.device, epoch)
        
        # writer.add_scalar('training loss', train_loss, epoch)
        
        # if (epoch+1)%config.save_every == 0:
        #     save_checkpoint(model, config.num_epoch, epoch, optimizer,train_loss, os.path.join(chk_path_folder,"save_at_epoch-"+str(epoch)+".chk"))
