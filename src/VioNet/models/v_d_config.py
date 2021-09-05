# '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/pytorch-i3d/models/rgb_imagenet.pt'
# '/media/david/datos/Violence DATA/VioNet_weights/pytorch_i3d/rgb_imagenet.pt
#'/content/drive/My Drive/VIOLENCE DATA/Pretrained_Models/pytorch_i3d/rgb_imagenet.pt'
# '/content/DATASETS/Pretrained_Models/rgb_imagenet.pt'
VD_CONFIG = {
    'backbone_name':'3dresnet', #i3d, 3dresnet
    'final_endpoint':'Mixed_4e', #Mixed_4e, so far, only for i3d
    'roi_layer_output':8,
    'roi_with_temporal_pool':True,
    'roi_spatial_scale':16,
    'fc_input_dim':528, #I3D-->528 for Mixed_4e, 832 for Mixed_4f, 3dResNet-->1024
    'roi_layer_type':'RoIAlign',
    'pretrained_backbone_model': '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/pytorch-i3d/models/rgb_imagenet.pt'
}

TWO_STREAM_CONFIG = {
    'backbone_name':'i3d', #i3d, 3dresnet
    'final_endpoint':'Mixed_4e', #Mixed_4e, so far, only for i3d
    'roi_layer_output':8,
    'roi_with_temporal_pool':True,
    'roi_spatial_scale':16,
    # 'fc_input_dim':528, #I3D-->528 for Mixed_4e, 832 for Mixed_4f, 3dResNet-->1024
    'roi_layer_type':'RoIAlign',
    'pretrained_backbone_model': '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/pytorch-i3d/models/rgb_imagenet.pt',
    '2d_backbone': 'resnet50',
    'base_out_layer': 'layer3',
    'num_trainable_layers': 3,
    'fc_input_dim': 1552
}

TWO_STREAM_CFAM_CONFIG_old = {
    'backbone_name':'i3d', #i3d, 3dresnet
    'final_endpoint':'Mixed_4e',#'Mixed_5b', #Mixed_4e, so far, only for i3d
    'with_roipool': True,
    'roi_layer_output':8,
    'roi_with_temporal_pool':True,
    'roi_spatial_scale':16,
    'roi_layer_type':'RoIAlign',
    'pretrained_backbone_model': '/media/david/datos/Violence DATA/VioNet_weights/pytorch_i3d/rgb_imagenet.pt',
    'freeze_3d': False,
    '2d_backbone': 'resnet50',
    'base_out_layer': 'layer3',
    'num_trainable_layers': 3,
    'CFAMBlock_in_channels': 528+1024,#528+1024,#528+1024, #832+2048
    'CFAMBlock_out_channels': 145, #1024
    'fc_input_dim': 8*8*145,#512#7105#145,#9280,
    
}

TWO_STREAM_CFAM_CONFIG = {
    'backbone_name':'i3d', #i3d, 3dresnet
    'final_endpoint':'Mixed_4e',#'Mixed_5b', #Mixed_4e, so far, only for i3d
    'with_roipool': True,
    'head': 'binary',
    'roi_layer_output':8,
    'roi_with_temporal_pool':True,
    'roi_spatial_scale':16,
    'roi_layer_type':'RoIAlign',
    'pretrained_backbone_model': '/media/david/datos/Violence DATA/VioNet_weights/pytorch_i3d/rgb_imagenet.pt',
    'freeze_3d': True,
    '2d_backbone': 'resnet50',
    'base_out_layer': 'layer3',
    'num_trainable_layers': 3,
    'CFAMBlock_in_channels': 528+1024,#528+1024,#528+1024, #832+2048
    'CFAMBlock_out_channels': 512, #1024
    'fc_input_dim': 8*8*512,#512#7105#145,#9280,
    'load_weigths': None#'/media/david/datos/Violence DATA/VioNet_pth/rwf-2000_model(TwoStreamVD_Binary_CFam)_head(binary)_stream(rgb)_cv(1)_epochs(100)_num_tubes(4)_framesXtube(16)_tub_sampl_rand(True)_optimizer(Adadelta)_lr(0.001)_note(TWO_STREAM_CFAM_CONFIG+RWF-2000-150frames-motion-maps2-centralframe-corrected)/save_at_epoch-49.chk'#'/media/david/datos/Violence DATA/VioNet_pth/rwf-2000_model(TwoStreamVD_Binary_CFam)_head(binary)_stream(rgb)_cv(1)_epochs(100)_num_tubes(4)_framesXtube(16)_tub_sampl_rand(True)_optimizer(Adadelta)_lr(0.001)_note(TWO_STREAM_CFAM_CONFIG+otherTrack)/save_at_epoch-39.chk'
    
}

TWO_STREAM_CFAM_NO_TUBE_CONFIG = {
    'backbone_name':'i3d', #i3d, 3dresnet
    'final_endpoint':'Mixed_5b',#'Mixed_5b', #Mixed_4e, so far, only for i3d
    'with_roipool': False,
    'roi_layer_output':8,
    'roi_with_temporal_pool':True,
    'roi_spatial_scale':16,
    'roi_layer_type':'RoIAlign',
    'pretrained_backbone_model': '/media/david/datos/Violence DATA/VioNet_weights/pytorch_i3d/rgb_imagenet.pt',
    'freeze_3d': False,
    '2d_backbone': 'resnet50',
    'base_out_layer': 'layer4',
    'num_trainable_layers': 4,
    'CFAMBlock_in_channels': 2048+832,#528+1024,#528+1024, #832+2048
    'CFAMBlock_out_channels': 1024, #1024
    'fc_input_dim': 7*7*1024,#512#7105#145,#9280,
    
}

TWO_STREAM_CFAM_SLOWRESNET_CONFIG = {
    'backbone_name':'3dresnet', #i3d, 3dresnet
    # 'final_endpoint':'Mixed_4e', #Mixed_4e, so far, only for i3d
    'with_roipool': True,
    'roi_layer_output':8,
    'roi_with_temporal_pool':True,
    'roi_spatial_scale':16,
    'roi_layer_type':'RoIAlign',
    # 'pretrained_backbone_model': '/media/david/datos/Violence DATA/VioNet_weights/pytorch_i3d/rgb_imagenet.pt',
    'freeze_3d': False,
    '2d_backbone': 'resnet50',
    'base_out_layer': 'layer3',
    'num_trainable_layers': 3,
    'CFAMBlock_in_channels': 1024+1024, #1024
    'CFAMBlock_out_channels': 145, #1024
    'fc_input_dim': 9280,#512#7105#145,#9280,
    
}

TWO_STREAM_REGRESSION_CONFIG = {
    'backbone_name':'i3d', #i3d, 3dresnet
    'final_endpoint':'Mixed_4e', #Mixed_4e, so far, only for i3d
    'roi_layer_output':8,
    'roi_with_temporal_pool':True,
    'roi_spatial_scale':16,
    'roi_layer_type':'RoIAlign',
    'pretrained_backbone_model': '/media/david/datos/Violence DATA/VioNet_weights/pytorch_i3d/rgb_imagenet.pt',
    '2d_backbone': 'resnet50',
    'base_out_layer': 'layer3',
    'num_trainable_layers': 0,
    'fc_input_dim': 1552
}