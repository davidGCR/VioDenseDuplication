import torch
import torch.nn as nn
import torch.nn.functional as F
from models.i3d import Unit3D, MaxPool3dSamePadding, InceptionModule
from models.roi_extractor_3d import SingleRoIExtractor3D

class InceptionI3d_Roi(nn.Module):
    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d_Roi, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)
        
        self.roi_op = SingleRoIExtractor3D(roi_layer_type='RoIAlign',
                                            featmap_stride=16,
                                            output_size=7, #8
                                            with_temporal_pool=False)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return
        # if self._final_endpoint == end_point:
        #     self.build()
        #     return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        # if self._final_endpoint == end_point: return
        
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        # if self._final_endpoint == end_point: return
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return
        # if self._final_endpoint == end_point:
        #     self.build()
        #     return

        self.avg_pool = nn.AvgPool3d(kernel_size=[4, 7, 7],
                                     stride=(1, 1, 1))

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        # if self._final_endpoint == end_point: return
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = 'Logits'
        # self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
        #                              stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()


    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        
    
    def build(self):
        # print("self.end_points.keys:", self.end_points.keys())
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x, bbox, num_tubes=0):
        batch, c, t, h, w = x.size()
        for end_point in self.VALID_ENDPOINTS: #torch.Size([4, 1024, 2, 7, 7])
            if end_point in self.end_points:
                if end_point == 'MaxPool3d_5a_2x2':
                    x, _ = self.roi_op(x, bbox)
                    # print('roi_pool', ':', x.size())
                    continue
                x = self._modules[end_point](x) # use _modules to work with dataparallel
                # print(end_point, ':', x.size())

        batch = int(batch/4)
        x = x.view(batch, 4, 1024, 4, 7 ,7)
        # print("view: ", x.size())
        x = x.max(dim=1).values
        # print("Before logits: ", x.size()) #torch.Size([16, 1024, 4, 7, 7])
        x = self.avg_pool(x) #torch.Size([4, 1024, 1, 1, 1])
        # print("After avg_pool: ", x.size())
        x = self.logits(self.dropout(x)) #torch.Size([4, 2, 1, 1, 1])
        # print("After logits: ", x.size())
        
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        logits = torch.squeeze(logits, dim=2) #added to train [batch, num_classes]
        # logits is batch X time X classes, which is what we want to work with
        return logits
        

if __name__=="__main__":
    print('------- i3d Roi --------')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    input = torch.rand(4,3,16,224,224).to(device) #for slowFAst backbone: 3x4x256x320, RWF-frames 224x224, RWF-video 640x360
    rois = torch.rand(4, 5).to(device)
    rois[0] = torch.tensor([0,  62.5481,  49.0223, 122.0747, 203.4146]).to(device)
    rois[1] = torch.tensor([1, 34, 14, 185, 177]).to(device)
    rois[2] = torch.tensor([2, 34, 14, 185, 177]).to(device)
    rois[3] = torch.tensor([3, 34, 14, 185, 177]).to(device)
    # i3d = InceptionI3d(2, in_channels=3, final_endpoint='Mixed_4f').to(device)
    i3d = InceptionI3d_Roi(num_classes=2, in_channels=3).to(device)
    output = i3d(input, rois)
    print("output({}): ".format(i3d._final_endpoint), output.size())
