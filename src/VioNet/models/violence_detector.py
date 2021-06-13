from typing import Dict
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

        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        #Classification Head
        self.detector = AnomalyDetector(input_dim=detector_input_dim)


    def __build_backbone__(self, backbone_name):
        if backbone_name == 'i3d':
            return  InceptionI3d(2, in_channels=3, final_endpoint=self.final_endpoint)
        else:
            return None

    
    def forward(self, x, bbox):
        #x: b,c,t,w,h
        x = self.backbone(x) #torch.Size([4, 528, 4, 14, 14])
        # central_bbox = self.__get_central_bbox__(tubelet_data)
        
        x = self.roi_op(x, bbox)
        x = self.temporal_pool(x)
        x = self.spatial_pool(x)
        return x

# from TubeletGeneration.tube_utils import JSON_2_tube
import numpy as np
import json
def JSON_2_tube(json_file):
    """
    """
    with open(json_file, "r") as read_file:
        decodedArray = json.load(read_file)
        # print("decoded Array:", type(decodedArray), len(decodedArray))
        
        for f in decodedArray:
            for i, box in  enumerate(f['boxes']):
                f['boxes'][i] = np.asarray(f['boxes'][i])
        # print(decodedArray[0])
        return decodedArray

def get_central_bbox(tubelet_data: Dict):
    tubelet = tubelet_data['boxes']
    if len(tubelet)>2:
        central_box = tubelet[int(len(tubelet)/2)]
    else:
        central_box = tubelet[0]

    central_box = np.insert(central_box[0:4], 0, 1).reshape(1,-1)
    central_box = torch.from_numpy(central_box).float() #remove score
    # id = 0
    # central_box = np.insert(central_box, 0, id)#add id at the begining
    # central_box = central_box.reshape(1,-1) #to shape (1,5)
    # central_box[0,0] = id
    return central_box

if __name__=='__main__':
    print('------- ViolenceDetector --------')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViolenceDetector(detector_input_dim=2048).to(device)
    

    input = torch.rand(1,3,16,224,224).to(device)
    
    tubes = JSON_2_tube('/media/david/datos/Violence DATA/Tubes/RWF-2000/train/Fight/_6-B11R9FJM_2.json')
    bbox = get_central_bbox(tubes[0]).to(device)
    
    print('central bbox:', bbox.size(), bbox)

    output = model(input, bbox)
    print('output: ', output.size())
