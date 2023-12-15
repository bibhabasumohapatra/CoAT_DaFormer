import sys

sys.path.append("/mnt/prj001/Bibhabasu_Mohapatra/RP_Gland_vRest/CoAT")

from coat import *
from daformer import *
from helper import *

import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

from config import NUM_CLASSES

class MixUpSample(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.mixing = nn.Parameter(torch.tensor(0.5))
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.mixing * F.interpolate(
            x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        ) + (1 - self.mixing) * F.interpolate(
            x, scale_factor=self.scale_factor, mode="nearest"
        )
        return x


class Net(nn.Module):
    def __init__(
        self,
        encoder=coat_lite_medium,
        decoder=daformer_conv3x3,
        encoder_cfg={},
        decoder_cfg={},
        num_classes=1,
    ):
        super(Net, self).__init__()
        decoder_dim = decoder_cfg.get("decoder_dim", 320)

        self.encoder = encoder
        encoder_dim = self.encoder.embed_dims
        # [64, 128, 320, 512]

        self.decoder = decoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
        )
        self.logit = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)
        self.mixup = MixUpSample()

    def forward(self, x):
        B, C, H, W = x.shape
        encoder = self.encoder(x)

        last, decoder = self.decoder(encoder)
        logits = self.logit(last)

        upsampled_logits = self.mixup(logits)

        return upsampled_logits


### encoder
class coat_parallel_small_plus1(CoaT):
    def __init__(self, **kwargs):
        super(coat_parallel_small_plus1, self).__init__(
            patch_size=4,
            embed_dims=[152, 320, 320, 320, 320],
            serial_depths=[2, 2, 2, 2, 2],
            parallel_depth=6,
            num_heads=8,
            mlp_ratios=[4, 4, 4, 4, 4],
            pretrain="coat_small_7479cf9b.pth",
            **kwargs
        )


def Model(num_classes=4):
    encoder = coat_lite_medium()
    checkpoint = "coat_lite_medium_384x384_f9129688.pth"
    checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    state_dict = checkpoint["model"]
    encoder.load_state_dict(state_dict, strict=False)

    net = Net(encoder=encoder, num_classes=num_classes).cuda()

    return net

from numerize import numerize
print(numerize.numerize(sum(p.numel() for p in Model().parameters())), " parameters")