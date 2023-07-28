from vit_pytorch import ViT
import torch
import torch.nn as nn
import torchvision.models as models
from einops.layers.torch import Rearrange

vit_encoder = ViT(
    image_size=14,  #H/8, W/8
    patch_size=1,
    dim = 128,
    depth=12,
    num_classes=2,
    heads=8,
    mlp_dim=10,
)
vit_encoder=vit_encoder.to('cuda')
# depth=12, head=8일때가 현재 최고설정
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )
class TransUnet_b5(nn.Module):
    def __init__(self):
        super(TransUnet_b5, self).__init__()
        self.backbone = models.efficientnet_b5(weights='EfficientNet_B5_Weights.DEFAULT')
        self.vit_flatten = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),
            nn.LayerNorm(128)
        )
        self.conv_vit_res = nn.Sequential(
            nn.Conv2d(128,64,3,1,1),
            nn.ReLU(inplace=True)
        )
        self.res_conv = nn.Sequential(
            nn.Conv2d(16,16,3,1,1),
            nn.ReLU(inplace=True)
        )
        self.seg_conv = nn.Conv2d(16, 1, 1)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(64 + 64, 64)
        self.dconv_up2 = double_conv(40 + 64, 32)
        self.dconv_up1 = double_conv(32 + 24, 16)

        self.conv_last = nn.Conv2d(16, 1, 1)
    def forward(self, x):
        x = self.backbone.features[0](x) #48,112,112
        out1 = self.backbone.features[1](x) #24,112,112
        out2 = self.backbone.features[2](out1) #40,56,56
        out3 = self.backbone.features[3](out2) #64,28,28
        out4 = self.backbone.features[4](out3) #128,14,14

        vit_out = self.vit_flatten(out4)
        vit_out = vit_encoder.transformer(vit_out)
        vit_out = vit_out.reshape(-1,128,14,14)
        vit_out = self.conv_vit_res(vit_out) #64,14,14
        up3 = self.upsample(vit_out)#64,28,28
        up3 = torch.cat([up3, out3], dim=1) #128,28,28
        up3 = self.dconv_up3(up3) #64,28,28

        up2 = self.upsample(up3) #64,56,56
        up2 = torch.cat([up2, out2], dim=1) #40+64,56,56
        up2 = self.dconv_up2(up2) #32,56,56

        up1 = self.upsample(up2) #32,112,112
        up1 = torch.cat([up1,out1],dim=1) #32+24,112,112
        up1 = self.dconv_up1(up1) #16,112,112

        up0 = self.upsample(up1)#16,224,224
        up0 = self.res_conv(up0)
        res = self.seg_conv(up0)
        return res
'''
from transUnet import TransUnet_b5
current setting: 224x224x3 image input
else requires changes in ViT image size parameter. 
'''