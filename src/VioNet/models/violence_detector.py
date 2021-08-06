import sys
sys.path.insert(1, '/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/AVSS2019/src/VioNet')
import torch
from torch import nn
from models.i3d import InceptionI3d
from models._3d_backbones import Backbone3DResNet, BackboneI3D
from models._2d_backbones import Backbone2DResNet
from models.roi_extractor_3d import SingleRoIExtractor3D
# from mmaction.models import SingleRoIExtractor3D

from models.anomaly_detector import AnomalyDetector
from torch.nn import functional as F
from models.v_d_config import *
from models.cfam import CFAMBlock 
# from i3d import InceptionI3d
# from roi_extractor_3d import SingleRoIExtractor3D
# from anomaly_detector import AnomalyDetector
REGRESSION = 'regression'
BINARY = 'binary'

class Learner(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0):
        super(Learner, self).__init__()
        # self.classifier = nn.Sequential(
        #     nn.Linear(input_dim, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.6),
        #     nn.Linear(512, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.6),
        #     nn.Linear(32, 1),
        #     nn.Sigmoid()
        # )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.drop_p = 0.6
        self.weight_init()
        self.vars = nn.ParameterList()

        for i, param in enumerate(self.classifier.parameters()):
            self.vars.append(param)

    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        x = F.linear(x, vars[0], vars[1])
        x = F.relu(x)
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[2], vars[3])
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[4], vars[5])
        return torch.sigmoid(x)

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars

class RoiPoolLayer(nn.Module):
    def __init__(self,roi_layer_type='RoIAlign',
                      roi_layer_output=8,
                      roi_with_temporal_pool=True,
                      roi_spatial_scale=16,
                      with_spatial_pool=True): #832):
        
        super(RoiPoolLayer, self).__init__()
        self.roi_op = SingleRoIExtractor3D(roi_layer_type=roi_layer_type,
                                            featmap_stride=roi_spatial_scale,
                                            output_size=roi_layer_output,
                                            with_temporal_pool=roi_with_temporal_pool)

        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.with_spatial_pool = with_spatial_pool
        
    
    def forward(self, x, bbox):
        #x: b,c,t,w,h
        batch, c, t, h, w = x.size()
        # print('before roipool: ', x.size(), ', bbox: ', bbox.size())
        x, _ = self.roi_op(x, bbox)
        # print('RoiPoolLayer after roipool: ', x.size())
        x = self.temporal_pool(x)
        # print('after temporal_pool: ', x.size())

        if self.with_spatial_pool:
            x = self.spatial_pool(x) #torch.Size([16, 528, 1, 1, 1]
            # print('after spatial_pool: ', x.size())
            x = x.view(x.size(0),-1)
        return x



class ViolenceDetectorRegression(nn.Module):
    def __init__(self,config=VD_CONFIG,
                      freeze=False
                      ):
        super(ViolenceDetectorRegression, self).__init__()
        self.config = config
        #Backbone
        self.final_endpoint = config['final_endpoint']
        self.backbone = BackboneI3D(
            config['final_endpoint'], 
            config['pretrained_backbone_model'],
            freeze=freeze)
        
        self.roi_layer = RoiPoolLayer(roi_layer_type=config['roi_layer_type'],
                            roi_layer_output=config['roi_layer_output'],
                            roi_with_temporal_pool=config['roi_with_temporal_pool'],
                            roi_spatial_scale=config['roi_spatial_scale'],
                            )
        self.fc = Learner(input_dim=config['fc_input_dim'])
    
    def forward(self, x, bbox, num_tubes=0):
        #x: b,c,t,w,h
        batch, c, t, h, w = x.size()
        batch = int(batch/num_tubes)
        x = self.backbone(x)
        # print('i3d out: ', x.size(), ' bbox: ',bbox.size())
        x = self.roi_layer(x, bbox)
        x = self.fc(x)
        return x

class TwoStreamVDRegression(nn.Module):
    def __init__(self,config=TWO_STREAM_REGRESSION_CONFIG,
                      freeze=False
                      ):
        super(TwoStreamVDRegression, self).__init__()
        self._3d_stream = BackboneI3D(
            config['final_endpoint'], 
            config['pretrained_backbone_model'],
            freeze)
        
        self._2d_stream = Backbone2DResNet(
            config['2d_backbone'],
            config['base_out_layer'],
            num_trainable_layers=config['num_trainable_layers'])
        
        self.roi_pool_3d = RoiPoolLayer(
            roi_layer_type=config['roi_layer_type'],
            roi_layer_output=config['roi_layer_output'],
            roi_with_temporal_pool=config['roi_with_temporal_pool'],
            roi_spatial_scale=config['roi_spatial_scale'],
            with_spatial_pool=True)
        
        self.roi_pool_2d = RoIAlign(output_size=config['roi_layer_output'],
                                    spatial_scale=config['roi_spatial_scale'],
                                    sampling_ratio=0,
                                    aligned=True
                                    )
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1,1))
        self.fc = Learner(input_dim=config['fc_input_dim'])
    
    def forward(self, x1, x2, bbox, num_tubes=0):
        batch, c, t, h, w = x1.size()
        batch = int(batch/num_tubes)

        x_3d = self._3d_stream(x1) #torch.Size([2, 528, 4, 14, 14])
        x_2d = self._2d_stream(x2) #torch.Size([2, 1024, 14, 14])

        x_3d = self.roi_pool_3d(x_3d,bbox)#torch.Size([8, 528])
        # print('3d after roipool: ', x_3d.size())

        x_2d = self.roi_pool_2d(x_2d,bbox)
        # print('2d after roipool: ', x_2d.size())

        x_2d = self.avg_pool_2d(x_2d)
        x_2d = torch.squeeze(x_2d)

        x = torch.cat((x_3d,x_2d), dim=1)
        x = self.fc(x)
        return x

class ViolenceDetectorBinary(nn.Module):
    def __init__(self,
        config=VD_CONFIG,  
        freeze=False):
        super(ViolenceDetectorBinary, self).__init__()
        self.config = config
        #Backbone
        self.final_endpoint = config['final_endpoint']
        self.backbone = self.__build_backbone__(config['backbone_name'])
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.roi_layer = RoiPoolLayer(roi_layer_type=config['roi_layer_type'],
                            roi_layer_output=config['roi_layer_output'],
                            roi_with_temporal_pool=config['roi_with_temporal_pool'],
                            roi_spatial_scale=config['roi_spatial_scale'],
                            )
        input_dim = config['fc_input_dim']
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 2),
            # nn.ReLU(),
            # nn.Dropout(0.6),
            # nn.Linear(128, 32),
            # nn.ReLU(),
            # nn.Dropout(0.6),
            # nn.Linear(32, 2),
            # nn.Sigmoid()
        )

    def __build_backbone__(self, backbone_name):
        if backbone_name == 'i3d':
            i3d = InceptionI3d(2, in_channels=3, final_endpoint=self.final_endpoint)
            #'/content/drive/My Drive/VIOLENCE DATA/Pretrained_Models/pytorch_i3d/rgb_imagenet.pt'
            if self.config['pretrained_backbone_model']:
                load_model_path = self.config['pretrained_backbone_model']#'/media/david/datos/Violence DATA/VioNet_weights/pytorch_i3d/rgb_imagenet.pt'
                state_dict = torch.load(load_model_path)
                i3d.load_state_dict(state_dict,  strict=False)
            return i3d
        elif backbone_name == '3dresnet':
            model = Backbone3DResNet()
            return model
        else:
            return None

    
    def forward(self, x, bbox, num_tubes=0):
        batch, c, t, h, w = x.size()
        batch = int(batch/num_tubes)
        x = self.backbone(x)
        x = self.roi_layer(x,bbox)
        x = x.view(batch, 4, -1)
        # print('before maxpool: ', x.size())
        x = x.max(dim=1).values
        # print('after maxpool: ', x.size())
            
        # print('i3d out: ', x.size(), ' bbox: ',bbox.size())
        x = self.classifier(x)
        return x

from torchvision.ops.roi_align import RoIAlign

class TwoStreamVD_Binary(nn.Module):
    def __init__(self, config=TWO_STREAM_CONFIG,
                        freeze=False):
        super(TwoStreamVD_Binary, self).__init__()
        self._3d_stream = BackboneI3D(
            config['final_endpoint'], 
            config['pretrained_backbone_model'])
        
        self._2d_stream = Backbone2DResNet(
            config['2d_backbone'],
            config['base_out_layer'],
            num_trainable_layers=config['num_trainable_layers'])
        
        self.roi_pool_3d = RoiPoolLayer(
            roi_layer_type=config['roi_layer_type'],
            roi_layer_output=config['roi_layer_output'],
            roi_with_temporal_pool=config['roi_with_temporal_pool'],
            roi_spatial_scale=config['roi_spatial_scale'],
            with_spatial_pool=False)
        
        self.roi_pool_2d = RoIAlign(output_size=config['roi_layer_output'],
                                    spatial_scale=config['roi_spatial_scale'],
                                    sampling_ratio=0,
                                    aligned=True
                                    )
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Linear(config['fc_input_dim'], 2),
            # nn.ReLU(),
            # nn.Dropout(0.6),
            # nn.Linear(128, 32),
            # nn.ReLU(),
            # nn.Dropout(0.6),
            # nn.Linear(32, 2),
            # nn.Sigmoid()
        )
        
    def forward(self, x1, x2, bbox, num_tubes=0):
        batch, c, t, h, w = x1.size()
        batch = int(batch/num_tubes)

        x_3d = self._3d_stream(x1) #torch.Size([2, 528, 4, 14, 14])
        x_2d = self._2d_stream(x2) #torch.Size([2, 1024, 14, 14])

       
        # print('output_3dbackbone: ', x_3d.size())
        # print('output_2dbackbone: ', x_2d.size())

        x_3d = self.roi_pool_3d(x_3d,bbox)#torch.Size([8, 528])
        print('3d after roipool: ', x_3d.size())

        x_2d = self.roi_pool_2d(x_2d,bbox)
        print('2d after roipool: ', x_2d.size())

        x_2d = self.avg_pool_2d(x_2d)
        x_2d = torch.squeeze(x_2d)

        x = torch.cat((x_3d,x_2d), dim=1)

        x = x.view(batch, 4, -1)
        x = x.max(dim=1).values

        x=self.classifier(x)

        return x

class TwoStreamVD_Binary_CFam(nn.Module):
    def __init__(self, config=TWO_STREAM_CFAM_CONFIG,
                        freeze=False):
        super(TwoStreamVD_Binary_CFam, self).__init__()
        self._3d_stream = BackboneI3D(
            config['final_endpoint'], 
            config['pretrained_backbone_model'])
        
        self._2d_stream = Backbone2DResNet(
            config['2d_backbone'],
            config['base_out_layer'],
            num_trainable_layers=config['num_trainable_layers'])
        
        self.roi_pool_3d = RoiPoolLayer(
            roi_layer_type=config['roi_layer_type'],
            roi_layer_output=config['roi_layer_output'],
            roi_with_temporal_pool=config['roi_with_temporal_pool'],
            roi_spatial_scale=config['roi_spatial_scale'],
            with_spatial_pool=False)
        
        self.roi_pool_2d = RoIAlign(output_size=config['roi_layer_output'],
                                    spatial_scale=config['roi_spatial_scale'],
                                    sampling_ratio=0,
                                    aligned=True
                                    )
        in_channels = 528+1024
        out_channels = 145                       
        self.CFAMBlock = CFAMBlock(in_channels, out_channels)
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Linear(config['fc_input_dim'], 2),
            # nn.ReLU(),
            # nn.Dropout(0.6),
            # nn.Linear(128, 32),
            # nn.ReLU(),
            # nn.Dropout(0.6),
            # nn.Linear(32, 2),
            # nn.Sigmoid()
        )
        
    def forward(self, x1, x2, bbox, num_tubes=0):
        batch, c, t, h, w = x1.size()
        batch = int(batch/num_tubes)

        x_3d = self._3d_stream(x1) #torch.Size([2, 528, 4, 14, 14])
        x_2d = self._2d_stream(x2) #torch.Size([2, 1024, 14, 14])

       
        # print('output_3dbackbone: ', x_3d.size())
        # print('output_2dbackbone: ', x_2d.size())

        x_3d = self.roi_pool_3d(x_3d,bbox)#torch.Size([8, 528])
        x_3d = torch.squeeze(x_3d)
        # print('3d after roipool: ', x_3d.size())

        x_2d = self.roi_pool_2d(x_2d,bbox)
        # print('2d after roipool: ', x_2d.size())

        x = torch.cat((x_3d,x_2d), dim=1) #torch.Size([8, 1552, 8, 8])
        x = self.CFAMBlock(x) #torch.Size([8, 145, 8, 8])

        # x = x.view(batch, 4, -1)
        x = x.view(batch, 4, 145, 8, 8)
        # x = self.avg_pool_2d(x)
        x = x.max(dim=1).values #torch.Size([2, 9280])
        # print('after max pool: ', x.size())
        x = self.avg_pool_2d(x)
        # print('after avgpool 2d: ', x.size())
        x = torch.squeeze(x)
        x=self.classifier(x)

        return x




if __name__=='__main__':
    
    print('------- ViolenceDetector --------')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = TwoStreamVD_Binary().to(device)
    model = TwoStreamVD_Binary_CFam().to(device)
    batch = 2
    tubes = 4
    input_1 = torch.rand(batch*tubes,3,16,224,224).to(device)

    input_2 = torch.rand(batch*tubes,3,224,224).to(device)

    rois = torch.rand(batch*tubes, 5).to(device)
    rois[0] = torch.tensor([0,  62.5481,  49.0223, 122.0747, 203.4146]).to(device)#torch.tensor([1, 14, 16, 66, 70]).to(device)
    rois[1] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    rois[2] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    rois[3] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    rois[4] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    rois[5] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    rois[6] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    rois[7] = torch.tensor([1, 34, 14, 85, 77]).to(device)

    output = model(input_1, input_2, rois, tubes)
    print('output: ', output.size())
    
    # regressor = ViolenceDetectorRegression().to(device)
    # input_1 = torch.rand(batch*tubes,3,16,224,224).to(device)
    # output = regressor(input_1, rois, tubes)
    # print('output: ', output.size())

