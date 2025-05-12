from math import pi

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import deform_conv2d
from mmdet.core import force_fp32
from mmdet.models.builder import ROI_EXTRACTORS
from mmdet.models.roi_heads.roi_extractors import BaseRoIExtractor


@ROI_EXTRACTORS.register_module()
class FAM3DRoIExtractor(BaseRoIExtractor):
    """FAM-3D enhanced RoI extractor with multi-level feature alignment.
    
    Args:
        offset_channels (int): Channels for offset calculation. Default: 64.
        use_atan_constraint (bool): Use arctangent for weight normalization. Default: True.
        finest_scale (int): Base scale for level mapping. Default: 56.
    """

    def __init__(self, 
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 offset_channels=64,
                 use_atan_constraint=True,
                 finest_scale=56):
        super().__init__(roi_layer, out_channels, featmap_strides)
        self.finest_scale = finest_scale
        self.use_atan = use_atan_constraint
        
        # FAM-3D specific components
        self.pyramid_offset = nn.Sequential(
            ConvModule(
                out_channels * 3,  # For lower/current/upper concatenation
                offset_channels,
                3,
                padding=1,
                norm_cfg=dict(type='GN')),
            nn.Conv2d(offset_channels, 1, 1))  # Weight prediction
        
        self.reg_offset_module = nn.Sequential(
            ConvModule(
                out_channels * 3,
                offset_channels,
                3,
                padding=1,
                norm_cfg=dict(type='GN')),
            nn.Conv2d(offset_channels, 4*2, 3, padding=1))  # 4 points * (x,y)

    def map_roi_levels(self, rois, num_levels):
        """Modified from SingleRoIExtractor with enhanced scale handling"""
        scale = torch.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.log2(scale / self.finest_scale + 1e-6)
        return target_lvls.clamp(min=0, max=num_levels-1).long()

    def _pyramid_feature_align(self, feats):
        """Core FAM-3D alignment logic"""
        aligned_feats = []
        for idx in range(len(feats)):
            # 1. Adjacent feature interpolation
            h, w = feats[idx].shape[2:]
            lower = F.interpolate(
                feats[idx-1], (h,w)) if idx > 0 else feats[idx]
            upper = F.interpolate(
                feats[idx+1], (h,w)) if idx < len(feats)-1 else feats[idx]
            
            # 2. Weight calculation with constraints
            fused_feat = torch.cat([lower, feats[idx], upper], dim=1)
            weight = self.pyramid_offset(fused_feat)
            
            # Activation constraints
            if self.use_atan:
                weight = weight.atan()/(pi/2)
                if idx == 0: weight = weight.clamp(min=0)
                if idx == len(feats)-1: weight = weight.clamp(max=0)
            else:
                if idx == 0: weight = weight.clamp(min=0)
                else: weight = weight.clamp(min=-1)
                if idx == len(feats)-1: weight = weight.clamp(max=0)
                else: weight = weight.clamp(max=1)
                
            # 3. Deformable convolution adaptation
            offset = self.reg_offset_module(fused_feat)
            aligned_feat = deform_conv2d(
                feats[idx], 
                offset,
                self.reg_offset_module[0].conv.weight.shape[-1],
                padding=(self.reg_offset_module[0].conv.weight.shape[-1]-1)//2
            )
            aligned_feats.append(aligned_feat)
        
        return aligned_feats

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Enhanced forward with FAM-3D alignment"""
        # 1. Perform FAM-3D feature alignment
        aligned_feats = self._pyramid_feature_align(feats)
        
        # 2. Original extraction logic with aligned features
        out_size = self.roi_layers[0].out_size
        num_levels = len(aligned_feats)
        roi_feats = feats[0].new_zeros(rois.size(0), self.out_channels, *out_size)
        
        if num_levels == 1:
            return self.roi_layers[0](aligned_feats[0], rois)
        
        # 3. Level-aware feature pooling
        target_lvls = self.map_roi_levels(rois, num_levels)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
            
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                roi_feats[inds] = self.roi_layers[i](
                    aligned_feats[i], rois[inds])
                
        return roi_feats
