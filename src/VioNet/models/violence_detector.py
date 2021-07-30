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
REGRESSION = 'regression'
BINARY = 'binary'

class RoiHead(nn.Module):
    def __init__(self,roi_layer_type='RoIAlign',
                      roi_layer_output=8,
                      roi_with_temporal_pool=True,
                      roi_spatial_scale=16,
                      fc_input_dim=528, #832
                      classifier=REGRESSION):
        
        super(RoiHead, self).__init__()
        self.roi_op = SingleRoIExtractor3D(roi_layer_type=roi_layer_type,
                                            featmap_stride=roi_spatial_scale,
                                            output_size=roi_layer_output,
                                            with_temporal_pool=roi_with_temporal_pool)

        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        self.classifier = classifier
        if self.classifier == REGRESSION:
            self.fc1 = nn.Linear(fc_input_dim, 128) #original was 512
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(0.6)

            self.fc2 = nn.Linear(128, 32)
            self.dropout2 = nn.Dropout(0.6)

            self.fc3 = nn.Linear(32, 1)

            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)
            nn.init.xavier_normal_(self.fc3.weight)
        else:
            self.fc1 = nn.Linear(fc_input_dim, 128)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(0.6)
            self.fc2 = nn.Linear(128, 32)
            self.dropout2 = nn.Dropout(0.6)
            self.fc3 = nn.Linear(32, 2)
            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)
            nn.init.xavier_normal_(self.fc3.weight)
    
    def forward(self, x, bbox):
        #x: b,c,t,w,h
        batch, c, t, h, w = x.size()
        # print('before roipool: ', x.size(), ', bbox: ', bbox.size())
        x, _ = self.roi_op(x, bbox)
        # print('after roipool: ', x.size())
        x = self.temporal_pool(x)
        # print('after temporal_pool: ', x.size())
        x = self.spatial_pool(x)
        # print('after spatial_pool: ', x.size())
        x = x.view(x.size(0),-1)

        # print('after x.view: ', x.size())
        # x = self.detector(x)
        if self.classifier == REGRESSION:
            x = self.dropout1(self.relu1(self.fc1(x)))
            x = self.dropout2(self.fc2(x))
            x = self.fc3(x)
        else:
            batch = int(batch/4)
            x = x.view(batch, 4, -1)
            # print('before maxpool: ', x.size())
            x = x.max(dim=1).values
            # print('after maxpool: ', x.size())
            x = self.dropout1(self.relu1(self.fc1(x)))
            x = self.dropout2(self.fc2(x))
            x = self.fc3(x)
            
        return x

from models.v_d_config import VD_CONFIG

class ViolenceDetector(nn.Module):
    def __init__(self,config=VD_CONFIG,
                      classifier=REGRESSION,
                      freeze=False
                      ):
        super(ViolenceDetector, self).__init__()
        self.config = config
        #Backbone
        self.final_endpoint = config['final_endpoint']
        self.backbone = self.__build_backbone__(config['backbone_name'])
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = RoiHead(roi_layer_type=config['roi_layer_type'],
                            roi_layer_output=config['roi_layer_output'],
                            roi_with_temporal_pool=config['roi_with_temporal_pool'],
                            roi_spatial_scale=config['roi_spatial_scale'],
                            fc_input_dim=config['fc_input_dim'],
                            classifier=classifier)
        self.classifier = classifier
        if self.classifier == REGRESSION:
            self.sigmoid = nn.Sigmoid()


    def __build_backbone__(self, backbone_name):
        if backbone_name == 'i3d':
            i3d = InceptionI3d(2, in_channels=3, final_endpoint=self.final_endpoint)
            #'/content/drive/My Drive/VIOLENCE DATA/Pretrained_Models/pytorch_i3d/rgb_imagenet.pt'
            load_model_path = self.config['pretrained_model']#'/media/david/datos/Violence DATA/VioNet_weights/pytorch_i3d/rgb_imagenet.pt'
            state_dict = torch.load(load_model_path)
            i3d.load_state_dict(state_dict,  strict=False)
            return i3d
        else:
            return None

    
    def forward(self, x, bbox, num_tubes=0):
        #x: b,c,t,w,h
        # x = self.backbone(x) #torch.Size([4, 528, 4, 14, 14])
        # x = self.head(x, bbox)
        # print('in: ', x.size())
        # print('boxes in: ', bbox)

        batch, c, t, h, w = x.size()
        batch = int(batch/4)
        x = self.backbone(x)
        # print('i3d out: ', x.size(), ' bbox: ',bbox.size())
        x = self.head(x, bbox)
        # print('head out: ', x.size())
        
        if self.classifier == REGRESSION:
            x = x.view(batch,4)
            x = x.max(dim=1).values
            x = self.sigmoid(x)
        

        # splits=torch.split(x, num_tubes.numpy().tolist())
        # out = []
        # for i in range(len(splits)):
        #   out.append(splits[i].max(dim=0)[0])
        # x = x.max(dim=0)
        
        return x

# class RLoss(torch.nn.Module):
#     def __init__(self, model, original_objective, lambdas=0.001):
#         super(RegularizedLoss, self).__init__()
#         self.lambdas = lambdas
#         self.model = model
#         self.objective = original_objective

#     def forward(self, y_pred, y_true):
#         # loss
#         # Our loss is defined with respect to l2 regularization, as used in the original keras code
#         fc1_params = torch.cat(tuple([x.view(-1) for x in self.model.fc1.parameters()]))
#         fc2_params = torch.cat(tuple([x.view(-1) for x in self.model.fc2.parameters()]))
#         fc3_params = torch.cat(tuple([x.view(-1) for x in self.model.fc3.parameters()]))

#         l1_regularization = self.lambdas * torch.norm(fc1_params, p=2)
#         l2_regularization = self.lambdas * torch.norm(fc2_params, p=2)
#         l3_regularization = self.lambdas * torch.norm(fc3_params, p=2)

#         return self.objective(y_pred, y_true) + l1_regularization + l2_regularization + l3_regularization

# def my_objective(y_pred, y_true):



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
