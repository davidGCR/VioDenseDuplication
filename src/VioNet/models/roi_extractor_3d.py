from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
# from mmcv.ops import RoIAlign, RoIPool
from torchvision.ops.roi_align import RoIAlign

# class SingleRoIExtractor3D(nn.Module):
#     """Extract RoI features from a single level feature map.
#     Args:
#         roi_layer_type (str): Specify the RoI layer type. Default: 'RoIAlign'.
#         featmap_stride (int): Strides of input feature maps. Default: 16.
#         output_size (int | tuple): Size or (Height, Width). Default: 16.
#         sampling_ratio (int): number of inputs samples to take for each
#             output sample. 0 to take samples densely for current models.
#             Default: 0.
#         pool_mode (str, 'avg' or 'max'): pooling mode in each bin.
#             Default: 'avg'.
#         aligned (bool): if False, use the legacy implementation in
#             MMDetection. If True, align the results more perfectly.
#             Default: True.
#         with_temporal_pool (bool): if True, avgpool the temporal dim.
#             Default: True.
#         with_global (bool): if True, concatenate the RoI feature with global
#             feature. Default: False.
#     Note that sampling_ratio, pool_mode, aligned only apply when roi_layer_type
#     is set as RoIAlign.
#     """

#     def __init__(self,
#                  roi_layer_type='RoIAlign',
#                  featmap_stride=16,
#                  output_size=16,
#                  sampling_ratio=0,
#                  pool_mode='avg',
#                  aligned=True,
#                  with_temporal_pool=True,
#                  with_global=False):
#         super().__init__()
#         self.roi_layer_type = roi_layer_type
#         assert self.roi_layer_type in ['RoIPool', 'RoIAlign']
#         self.featmap_stride = featmap_stride
#         self.spatial_scale = 1. / self.featmap_stride

#         self.output_size = output_size
#         self.sampling_ratio = sampling_ratio
#         self.pool_mode = pool_mode
#         self.aligned = aligned

#         self.with_temporal_pool = with_temporal_pool
#         self.with_global = with_global

#         if self.roi_layer_type == 'RoIPool':
#             self.roi_layer = RoIPool(self.output_size, self.spatial_scale)
#         else:
#             self.roi_layer = RoIAlign(
#                 self.output_size,
#                 self.spatial_scale,
#                 sampling_ratio=self.sampling_ratio,
#                 pool_mode=self.pool_mode,
#                 aligned=self.aligned)
#         self.global_pool = nn.AdaptiveAvgPool2d(self.output_size)

#     def init_weights(self):
#         pass

#     # The shape of feat is N, C, T, H, W
#     def forward(self, feat, rois):
#         if not isinstance(feat, tuple):
#             feat = (feat, )
#         if len(feat) >= 2:
#             assert self.with_temporal_pool
#         if self.with_temporal_pool:
#             feat = [torch.mean(x, 2, keepdim=True) for x in feat]
#         feat = torch.cat(feat, axis=1)
#         # print('_feat:',feat.size())

#         roi_feats = []
#         for t in range(feat.size(2)):
#             frame_feat = feat[:, :, t].contiguous()
#             # print('frame_feat:',frame_feat.size())
#             roi_feat = self.roi_layer(frame_feat, rois)
#             if self.with_global:
#                 global_feat = self.global_pool(frame_feat.contiguous())
#                 inds = rois[:, 0].type(torch.int64)
#                 global_feat = global_feat[inds]
#                 roi_feat = torch.cat([roi_feat, global_feat], dim=1)
#                 roi_feat = roi_feat.contiguous()
#             roi_feats.append(roi_feat)

#         return torch.stack(roi_feats, dim=2)

class SingleRoIExtractor3D(nn.Module):
    """Extract RoI features from a single level feature map.

    Args:
        roi_layer_type (str): Specify the RoI layer type. Default: 'RoIAlign'.
        featmap_stride (int): Strides of input feature maps. Default: 16.
        output_size (int | tuple): Size or (Height, Width). Default: 16.
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
            Default: 0.
        pool_mode (str, 'avg' or 'max'): pooling mode in each bin.
            Default: 'avg'.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
            Default: True.
        with_temporal_pool (bool): if True, avgpool the temporal dim.
            Default: True.
        with_global (bool): if True, concatenate the RoI feature with global
            feature. Default: False.

    Note that sampling_ratio, pool_mode, aligned only apply when roi_layer_type
    is set as RoIAlign.
    """

    def __init__(self,
                 roi_layer_type='RoIAlign',
                 featmap_stride=16,
                 output_size=16,
                 sampling_ratio=0,
                 pool_mode='avg',
                 aligned=True,
                 with_temporal_pool=True,
                 temporal_pool_mode='avg',
                 with_global=False):
        super().__init__()
        self.roi_layer_type = roi_layer_type
        assert self.roi_layer_type in ['RoIPool', 'RoIAlign']
        self.featmap_stride = featmap_stride
        self.spatial_scale = 1. / self.featmap_stride

        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.pool_mode = pool_mode
        self.aligned = aligned

        self.with_temporal_pool = with_temporal_pool
        self.temporal_pool_mode = temporal_pool_mode

        self.with_global = with_global

        if self.roi_layer_type == 'RoIPool':
            self.roi_layer = RoIPool(self.output_size, self.spatial_scale)
        else:
            # self.roi_layer = RoIAlign(
            #     self.output_size,
            #     self.spatial_scale,
            #     sampling_ratio=self.sampling_ratio,
            #     pool_mode=self.pool_mode,
            #     aligned=self.aligned)
            self.roi_layer = RoIAlign(output_size=self.output_size,
                                        spatial_scale=self.spatial_scale,
                                        sampling_ratio=self.sampling_ratio,
                                        aligned=self.aligned)

        self.global_pool = nn.AdaptiveAvgPool2d(self.output_size)

    def init_weights(self):
        pass

    # The shape of feat is N, C, T, H, W
    def forward(self, feat, rois):
        if not isinstance(feat, tuple):
            feat = (feat, )

        if len(feat) >= 2:
            maxT = max([x.shape[2] for x in feat])
            max_shape = (maxT, ) + feat[0].shape[3:]
            # resize each feat to the largest shape (w. nearest)
            feat = [F.interpolate(x, max_shape).contiguous() for x in feat]

        if self.with_temporal_pool:
            if self.temporal_pool_mode == 'avg':
                feat = [torch.mean(x, 2, keepdim=True) for x in feat]
            elif self.temporal_pool_mode == 'max':
                feat = [torch.max(x, 2, keepdim=True)[0] for x in feat]
            else:
                raise NotImplementedError

        feat = torch.cat(feat, axis=1).contiguous()

        # print('feat:',feat.size(), feat.device)

        roi_feats = []
        # rois[:,0] = rois[:,0]-rois[0,0] 
        # print('rois: ', rois) 
        for t in range(feat.size(2)):
            frame_feat = feat[:, :, t].contiguous()
            # print('frame feat t:',frame_feat.size(), frame_feat.device)
            # print('rois:',rois.size(), rois.device)
            roi_feat = self.roi_layer(frame_feat, rois)
            # print('roi_feat:',roi_feat.size(), roi_feat.device)
            if self.with_global:
                global_feat = self.global_pool(frame_feat.contiguous())
                inds = rois[:, 0].type(torch.int64)
                global_feat = global_feat[inds]
                roi_feat = torch.cat([roi_feat, global_feat], dim=1)
                roi_feat = roi_feat.contiguous()
            roi_feats.append(roi_feat)

        return torch.stack(roi_feats, dim=2), feat
        # return torch.stack(roi_feats, dim=2)

if __name__=='__main__':
    print('-----Roi3D------')
    # type='SingleRoIExtractor3D',
    # roi_layer_type='RoIAlign',
    # output_size=8,
    # with_temporal_pool=True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_map = torch.rand(3, 3, 4, 14, 14).to(device)

    num_rois = 3
    rois = torch.rand(num_rois, 5).to(device)
    rois[0] = torch.tensor([0,  62.5481,  49.0223, 122.0747, 203.4146]).to(device)#torch.tensor([1, 14, 16, 66, 70]).to(device)
    rois[1] = torch.tensor([0, 34, 14, 85, 77]).to(device)
    rois[2] = torch.tensor([1, 100, 126, 122, 130]).to(device)

    roi_op = SingleRoIExtractor3D(roi_layer_type='RoIAlign',
                                    output_size=8,
                                    featmap_stride=16,
                                    sampling_ratio=0,
                                    with_temporal_pool=True).to(device)
                # roi_layer_type='RoIAlign',
                #  featmap_stride=16,
                #  output_size=16,
                #  sampling_ratio=0,
                #  pool_mode='avg',
                #  aligned=True,
                #  with_temporal_pool=True,
                #  temporal_pool_mode='avg',
                #  with_global=False
    out,_ = roi_op(feature_map, rois)
    print('out:', out.size())

    # for i in range(out.size(0)):
    #     print(out[i], out[i].size())

    # tmp_pool_avg = nn.AdaptiveAvgPool3d((1, None, None)) #torch.Size([1, 2048, 1, 8, 8])
    # tmp_pool_max = nn.AdaptiveMaxPool3d((1, None, None))
    # sp_pool_avg = nn.AdaptiveAvgPool3d((None, 1, 1))
    # sp_pool_max = nn.AdaptiveMaxPool3d((None, 1, 1))

    # fake_map = torch.rand(16, 2048, 4, 8, 8).to(device)
    # out = tmp_pool_avg(fake_map)
    # print('tmp_pool out:', out.size())

    # out = sp_pool_avg(out)
    # print('spatial_pool out:', out.size())

    # out = out.view(out.size(0),-1)
    # print('view out:', out.size())

