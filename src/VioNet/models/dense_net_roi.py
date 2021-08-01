import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from densenet import _DenseBlock, _Transition
from roi_extractor_3d import SingleRoIExtractor3D

class DenseNet_Roi(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 sample_size,
                 sample_duration,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000):

        super(DenseNet_Roi, self).__init__()

        # self.sample_size = sample_size
        self.sample_duration = sample_duration

        self.endpoints = {}
        # First convolution
        end_point = 'conv0'
        self.endpoints[end_point] =  nn.Conv3d(3,
                           num_init_features,
                           kernel_size=7,
                           stride=(1, 2, 2),
                           padding=(3, 3, 3),
                           bias=False)
        end_point = 'norm0'
        self.endpoints[end_point] = nn.BatchNorm3d(num_init_features)
        end_point = 'relu0'
        self.endpoints[end_point] = nn.ReLU(inplace=True)
        end_point = 'pool0'
        self.endpoints[end_point] = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            # self.features.add_module('denseblock%d' % (i + 1), block)
            end_point = 'denseblock%d' % (i + 1)
            self.endpoints[end_point] = block
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                # self.features.add_module('transition%d' % (i + 1), trans)
                end_point = 'transition%d' % (i + 1)
                self.endpoints[end_point] = trans
                num_features = num_features // 2

        # Final batch norm
        end_point = 'norm5'
        self.endpoints[end_point] = nn.BatchNorm3d(num_features)
        # self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        self.roi_op = SingleRoIExtractor3D(roi_layer_type='RoIAlign',
                                            featmap_stride=16,
                                            output_size=7, #8
                                            with_temporal_pool=False)

        self.build()
        # Linear layer
        # end_point = 'classifier'
        # self.endpoints[end_point] = nn.Linear(num_features, num_classes)
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def build(self):
        for k in self.endpoints.keys():
            self.add_module(k, self.endpoints[k])

    def forward(self, x, box):
        batch, c, t, h, w = x.size()
        # features = self.features(x, bbox)
        for end_point in self.endpoints.keys():
            if end_point == 'denseblock2':
                x, _ = self.roi_op(x,box)
                print('{}: {}'.format('roi_pool', x.size()))
            x = self._modules[end_point](x)
            print('{}: {}'.format(end_point, x.size()))

        batch = int(batch/4)
        x = x.view(batch, 4, 1024, 2, 3 ,3)
        # print("view: ", x.size())
        x = x.max(dim=1).values

        out = F.relu(x, inplace=True)

        out = F.adaptive_avg_pool3d(out, (1, 1, 1)).view(x.size(0), -1)

        out = self.classifier(out)
        return out
# DenseNet Lean
def densenet88_roi(**kwargs):
    model = DenseNet_Roi(num_init_features=64,
                     growth_rate=32,
                     block_config=(6, 12, 24),
                     **kwargs)
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.rand(16,3,16,224,224).to(device)
    rois = torch.rand(16, 5).to(device)
    for i in range(input.size(0)):
        rois[i] = torch.tensor([i,  62.5481,  49.0223, 122.0747, 203.4146]).to(device)
    
    model = densenet88_roi  (num_classes=2,
                       sample_size=112,
                       sample_duration=12).to(device)
    # print(model)
    output = model(input, rois)
    print('out: ',output.size(), output)