# '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/pytorch-i3d/models/rgb_imagenet.pt'
# '/media/david/datos/Violence DATA/VioNet_weights/pytorch_i3d/rgb_imagenet.pt
#'/content/drive/My Drive/VIOLENCE DATA/Pretrained_Models/pytorch_i3d/rgb_imagenet.pt'
VD_CONFIG = {
    'backbone_name':'3dresnet', #i3d, 3dresnet
    'final_endpoint':'Mixed_4e', #Mixed_4e, so far, only for i3d
    'roi_layer_output':8,
    'roi_with_temporal_pool':True,
    'roi_spatial_scale':16,
    'fc_input_dim':528, #I3D-->528 for Mixed_4e, 832 for Mixed_4f, 3dResNet-->1024
    'roi_layer_type':'RoIAlign',
    'pretrained_backbone_model': '/media/david/datos/Violence DATA/VioNet_weights/pytorch_i3d/rgb_imagenet.pt'
}

TWO_STREAM_CONFIG = {
    'backbone_name':'i3d', #i3d, 3dresnet
    'final_endpoint':'Mixed_4e', #Mixed_4e, so far, only for i3d
    'roi_layer_output':8,
    'roi_with_temporal_pool':True,
    'roi_spatial_scale':16,
    'fc_input_dim':528, #I3D-->528 for Mixed_4e, 832 for Mixed_4f, 3dResNet-->1024
    'roi_layer_type':'RoIAlign',
    'pretrained_backbone_model': '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/pytorch-i3d/models/rgb_imagenet.pt',
    '2d_backbone': 'resnet50',
    'base_out_layer': 'layer3',
    'num_trainable_layers': 3,
    'fc_input_dim': 1552
}