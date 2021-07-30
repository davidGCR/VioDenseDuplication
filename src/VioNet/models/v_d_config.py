VD_CONFIG = {
    'backbone_name':'i3d',
    'final_endpoint':'Mixed_4e', #Mixed_4e
    'roi_layer_output':8,
    'roi_with_temporal_pool':True,
    'roi_spatial_scale':16,
    'fc_input_dim':528, #528 for Mixed_4e, 832 for Mixed_4f
    'roi_layer_type':'RoIAlign',
    'pretrained_model': '/media/david/datos/Violence DATA/VioNet_weights/pytorch_i3d/rgb_imagenet.pt'#'/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/pytorch-i3d/models/rgb_imagenet.pt'#'/content/drive/My Drive/VIOLENCE DATA/Pretrained_Models/pytorch_i3d/rgb_imagenet.pt'
}