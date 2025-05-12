from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .hourglass import HourglassNet
from .hrnet import HRNet
from .regnet import RegNet
from .res2net import Res2Net
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .csp_darknet import CSPDarknet
from .darknet import Darknet
from .swin import SwinTransformer
from .vit_win_rvsa_wsz7 import ViT_Win_RVSA_V3_WSZ7
from .vit_win_rvsa_kvdiff_wsz7 import ViT_Win_RVSA_V3_KVDIFF_WSZ7
from .vitae_nc_win_rvsa_wsz7 import ViTAE_NC_Win_RVSA_V3_WSZ7
from .vitae_nc_win_rvsa_kvdiff_wsz7 import ViTAE_NC_Win_RVSA_V3_KVDIFF_WSZ7

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'Res2Net',
    'HourglassNet', 'DetectoRS_ResNet', 'DetectoRS_ResNeXt',
    'ViT_Win_RVSA_V3_WSZ7', 'ViT_Win_RVSA_V3_KVDIFF_WSZ7',
    'ViTAE_NC_Win_RVSA_V3_WSZ7', 'ViTAE_NC_Win_RVSA_V3_KVDIFF_WSZ7',
    'Darknet', 'CSPDarknet', 'SwinTransformer'
]
