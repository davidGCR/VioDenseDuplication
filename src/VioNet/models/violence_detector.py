import torch
from torch import nn
from i3d import InceptionI3d
from roi_extractor_3d import SingleRoIExtractor3D
from anomaly_detector import AnomalyDetector


class ViolenceDetector(nn.Module):
    def __init__(self,backbone_name='i3d',
                      final_endpoint='Mixed_4e',
                      roi_layer_type='RoIAlign',
                      roi_layer_output=8,
                      roi_with_temporal_pool=True,
                      roi_spatial_scale=16,
                      detector_input_dim=2048
                      ):
        super(ViolenceDetector, self).__init__()
        #Backbone
        self.final_endpoint = final_endpoint
        self.backbone = self.__build_backbone__(backbone_name)
        
        #Roi_pool
        self.roi_op = SingleRoIExtractor3D(roi_layer_type=roi_layer_type, featmap_stride=roi_spatial_scale, output_size=roi_layer_output, with_temporal_pool=roi_with_temporal_pool)

        #Classification Head
        self.detector = AnomalyDetector(input_dim=detector_input_dim)


    def __build_backbone__(self, backbone_name):
        if backbone_name == 'i3d':
            return  InceptionI3d(2, in_channels=3, final_endpoint=self.final_endpoint)
        else:
            return None
    
    def __get_central_bbox__(self, tubelet):
        
    
    def forward(self, x, tubelet):
        #x: b,c,t,w,h
        video_feature = self.backbone(x)
        roi_feature = self.roi_op(video_feature, central_bbox)
