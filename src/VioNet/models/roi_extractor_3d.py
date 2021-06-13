import torch
import torch.nn as nn
from mmcv.ops import RoIAlign, RoIPool

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
        self.with_global = with_global

        if self.roi_layer_type == 'RoIPool':
            self.roi_layer = RoIPool(self.output_size, self.spatial_scale)
        else:
            self.roi_layer = RoIAlign(
                self.output_size,
                self.spatial_scale,
                sampling_ratio=self.sampling_ratio,
                pool_mode=self.pool_mode,
                aligned=self.aligned)
        self.global_pool = nn.AdaptiveAvgPool2d(self.output_size)

    def init_weights(self):
        pass

    # The shape of feat is N, C, T, H, W
    def forward(self, feat, rois):
        if not isinstance(feat, tuple):
            feat = (feat, )
        if len(feat) >= 2:
            assert self.with_temporal_pool
        if self.with_temporal_pool:
            feat = [torch.mean(x, 2, keepdim=True) for x in feat]
        feat = torch.cat(feat, axis=1)
        print('_feat:',feat.size())

        roi_feats = []
        for t in range(feat.size(2)):
            frame_feat = feat[:, :, t].contiguous()
            print('frame_feat:',frame_feat.size())
            roi_feat = self.roi_layer(frame_feat, rois)
            if self.with_global:
                global_feat = self.global_pool(frame_feat.contiguous())
                inds = rois[:, 0].type(torch.int64)
                global_feat = global_feat[inds]
                roi_feat = torch.cat([roi_feat, global_feat], dim=1)
                roi_feat = roi_feat.contiguous()
            roi_feats.append(roi_feat)

        return torch.stack(roi_feats, dim=2)

if __name__=='__main__':
    # type='SingleRoIExtractor3D',
    # roi_layer_type='RoIAlign',
    # output_size=8,
    # with_temporal_pool=True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    feature_map = torch.rand(1, 2048, 4, 16, 20).to(device)

    rois = torch.rand(1, 5).to(device)
    rois[0] = torch.tensor([1, 14, 16, 66, 70]).to(device)
    # rois[1] = torch.tensor([2, 34, 14, 85, 77]).to(device)
    # rois[2] = torch.tensor([3, 100, 126, 122, 130]).to(device)

    roi_op = SingleRoIExtractor3D(roi_layer_type='RoIAlign', output_size=8, with_temporal_pool=True).to(device)
    out=roi_op(feature_map, rois)
    print('out:', out.size())

    tmp_pool_avg = nn.AdaptiveAvgPool3d((1, None, None)) #torch.Size([1, 2048, 1, 8, 8])
    tmp_pool_max = nn.AdaptiveMaxPool3d((1, None, None))
    sp_pool_avg = nn.AdaptiveAvgPool3d((None, 1, 1))
    sp_pool_max = nn.AdaptiveMaxPool3d((None, 1, 1))

    fake_map = torch.rand(16, 2048, 4, 8, 8).to(device)
    out = tmp_pool_avg(fake_map)
    print('tmp_pool out:', out.size())

    out = sp_pool_avg(out)
    print('spatial_pool out:', out.size())

    out = out.view(out.size(0),-1)
    print('view out:', out.size())

