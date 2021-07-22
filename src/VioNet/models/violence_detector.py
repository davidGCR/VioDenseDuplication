from typing import Dict
import torch
from torch import nn
from models.i3d import InceptionI3d
from models.roi_extractor_3d import SingleRoIExtractor3D
# from mmaction.models import SingleRoIExtractor3D

from models.anomaly_detector import AnomalyDetector
# from i3d import InceptionI3d
# from roi_extractor_3d import SingleRoIExtractor3D
# from anomaly_detector import AnomalyDetector

class RoiHead(nn.Module):
    def __init__(self,roi_layer_type='RoIAlign',
                      roi_layer_output=8,
                      roi_with_temporal_pool=True,
                      roi_spatial_scale=16,
                      fc_input_dim=528):
        
        super(RoiHead, self).__init__()
        self.roi_op = SingleRoIExtractor3D(roi_layer_type=roi_layer_type,
                                            featmap_stride=roi_spatial_scale,
                                            output_size=roi_layer_output,
                                            with_temporal_pool=roi_with_temporal_pool)

        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.detector = nn.Sequential(
            nn.Linear(fc_input_dim, 128), #original was 512, 528
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, 32),
            nn.Dropout(0.6),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, bbox):
        #x: b,c,t,w,h
        x, _ = self.roi_op(x, bbox)
        # print('X type after ROIAling:', type(x), x.device)
        # print('X after ROIAling:', x.size())
        x = self.temporal_pool(x)
        x = self.spatial_pool(x)
        x = x.view(x.size(0),-1)
        # print('X view:', x.size())
        x = self.detector(x)
        # x = self.fc1(x)
        return x



class ViolenceDetector(nn.Module):
    def __init__(self,backbone_name='i3d',
                      final_endpoint='Mixed_4e',
                      roi_layer_type='RoIAlign',
                      roi_layer_output=8,
                      roi_with_temporal_pool=True,
                      roi_spatial_scale=16,
                      fc_input_dim=2048
                      ):
        super(ViolenceDetector, self).__init__()
        #Backbone
        self.final_endpoint = final_endpoint
        self.backbone = self.__build_backbone__(backbone_name)

        self.head = RoiHead(roi_layer_type=roi_layer_type,
                            roi_layer_output=roi_layer_output,
                            roi_with_temporal_pool=roi_with_temporal_pool,
                            roi_spatial_scale=roi_spatial_scale,
                            fc_input_dim=fc_input_dim)
        


    def __build_backbone__(self, backbone_name):
        if backbone_name == 'i3d':
            i3d = InceptionI3d(2, in_channels=3, final_endpoint=self.final_endpoint)
            #'/content/drive/My Drive/VIOLENCE DATA/Pretrained_Models/pytorch_i3d/rgb_imagenet.pt'
            load_model_path = '/media/david/datos/Violence DATA/VioNet_weights/pytorch_i3d/rgb_imagenet.pt'
            state_dict = torch.load(load_model_path)
            i3d.load_state_dict(state_dict,  strict=False)
            return i3d
        else:
            return None

    
    def forward(self, x, bbox, num_tubes=0):
        #x: b,c,t,w,h
        # x = self.backbone(x) #torch.Size([4, 528, 4, 14, 14])
        # x = self.head(x, bbox)
        # batch, c, t, h, w = x.size()
        x = self.backbone(x)
        # scores = []
        # for j in range(num_tubes.size(0)): # iterate over videos into a batch
        #     n_tubes = num_tubes[j]
        #     video_batch = x[num_tubes[j-1]:j*n_tubes + n_tubes, :, :, :, :]
        #     out = self.backbone(x)
        #     # get the max score for each video
        #     instance_max_score = out.max(dim=0)[0]
        # #     scores.append(instance_max_score.item())
        
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = ViolenceDetector(detector_input_dim=528).to(device)
    # print(model)
    
    tubes_num = 4
    input = torch.rand(tubes_num,3,16,224,224).to(device)
    
    tubes = JSON_2_tube('/media/david/datos/Violence DATA/Tubes/RWF-2000/train/Fight/_6-B11R9FJM_2.json')
    bbox = get_central_bbox(tubes[0]).to(device)
    
    print('central bbox:', bbox.size(), bbox)

    output = model(input, bbox)
    print('output: ', output.size())
