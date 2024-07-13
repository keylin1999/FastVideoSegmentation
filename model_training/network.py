import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from model_training.unets import AttU_Net, U_Net
from model_training.unetpp import NestedUNet
from model_training.test import VitSeg

from typing import Optional, List
import torch

def interpolate_groups(g: torch.Tensor, ratio: Optional[List[float]] = (1.0, 1.0), mode: str ='bilinear', align_corners: Optional[bool] = None):
    g = F.interpolate(g,
                scale_factor=ratio, mode=mode, align_corners=align_corners)
    return g

def upsample_groups(g: torch.Tensor, ratio: Optional[List[float]]=(2.0, 2.0), mode: str='bilinear', align_corners: Optional[bool]=False):
    return interpolate_groups(g, ratio, mode, align_corners)

class GroupResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        if in_dim == out_dim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
 
    def forward(self, g):
        out_g = self.conv1(F.relu(g))
        out_g = self.conv2(F.relu(out_g))
        
        if self.downsample is not None:
            g = self.downsample(g)

        return out_g + g


class Encoder(nn.Module):
    def __init__(self, model, depth=4) -> None:
        super().__init__()
        model = model(pretrained=True)

        self.conv1 = model.conv1        
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(model.__getattr__(f"layer{i+1}"))

    def forward(self, f):
        x = self.conv1(f) 
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        fd = [] # f4, f8, f16, f32

        for layer in self.layers:
            x = layer(x)
            fd.append(x)

        fd.reverse()

        return fd

class Decoder(nn.Module):
    def __init__(self, model, depth):
        super().__init__()
        channels = [1024, 512, 256, 128, 64, 32]
        if model == resnet18 or model == resnet34:
            channels = [512, 256, 128, 64, 32]
        else:
            channels = [2048, 1024, 512, 256, 128]
        channels = channels[4-depth:]

        self.grbs = nn.ModuleList()
        self.depth = depth
        for i in range(depth):
            self.grbs.append(GroupResBlock(channels[i], channels[i+1]))
        self.last = GroupResBlock(channels[i+1], 2)

    def forward(self, fd: List[torch.Tensor]):
        assert len(fd) == len(self.grbs)

        cat = fd[0]
        # batch, channel, h, w

        for i, grb in enumerate(self.grbs):
            upcat = upsample_groups(cat, ratio=(2.0, 2.0))
            upcat = grb(upcat)
            if i < self.depth - 1:
                upcat = upcat + fd[i+1]
            cat = upcat

        upcat = upsample_groups(upcat, ratio=(2.0, 2.0))
        upcat = self.last(upcat)

        return upcat.squeeze(1)
    
class UNet(nn.Module):
    def __init__(self, model, depth=4) -> None:
        super().__init__()
        self.pred_encoder = Encoder(model, depth=depth)
        self.decoder = Decoder(model, depth=depth)

    def forward(self, pred_image: torch.Tensor):
        pd = self.pred_encoder(torch.cat([pred_image], dim=1))

        logit = self.decoder(pd)
        prob = F.softmax(logit, 1)

        return prob, logit


    def load_weights(self, src_dict):
        self.load_state_dict(src_dict)

class Resnet18(UNet):
    def __init__(self, model=resnet18, depth=4) -> None:
        super().__init__(model, depth=depth)

class Resenet18Depth4(Resnet18):
    def __init__(self, depth=4) -> None:
        super().__init__(depth=depth)

class Resenet18Depth3(Resnet18):
    def __init__(self, depth=3) -> None:
        super().__init__(depth=depth)

class Resenet18Depth2(Resnet18):
    def __init__(self, depth=2) -> None:
        super().__init__(depth=depth)

class Resenet18Depth1(Resnet18):
    def __init__(self, depth=1) -> None:
        super().__init__(depth=depth)

# ----------------------------

class Resnet34(UNet):
    def __init__(self, model=resnet34, depth=4) -> None:
        super().__init__(model, depth=depth)

class Resenet34Depth4(Resnet34):
    def __init__(self, depth=4) -> None:
        super().__init__(depth=depth)

class Resenet34Depth3(Resnet34):
    def __init__(self, depth=3) -> None:
        super().__init__(depth=depth)

class Resenet34Depth2(Resnet34):
    def __init__(self, depth=2) -> None:
        super().__init__(depth=depth)

class Resenet34Depth1(Resnet34):
    def __init__(self, depth=1) -> None:
        super().__init__(depth=depth)

# ----------------------------

class Resnet50(UNet):
    def __init__(self, model=resnet50, depth=4) -> None:
        super().__init__(model, depth=depth)

class Resenet50Depth4(Resnet50):
    def __init__(self, depth=4) -> None:
        super().__init__(depth=depth)

class Resenet50Depth3(Resnet50):
    def __init__(self, depth=3) -> None:
        super().__init__(depth=depth)

class Resenet50Depth2(Resnet50):
    def __init__(self, depth=2) -> None:
        super().__init__(depth=depth)

class Resenet50Depth1(Resnet50):
    def __init__(self, depth=1) -> None:
        super().__init__(depth=depth)

# ----------------------------

class Resnet101(UNet):
    def __init__(self, model=resnet101, depth=4) -> None:
        super().__init__(model, depth=depth)

class Resenet101Depth4(Resnet101):
    def __init__(self, depth=4) -> None:
        super().__init__(depth=depth)

class Resenet101Depth3(Resnet101):
    def __init__(self, depth=3) -> None:
        super().__init__(depth=depth)

class Resenet101Depth2(Resnet101):
    def __init__(self, depth=2) -> None:
        super().__init__(depth=depth)

class Resenet101Depth1(Resnet101):
    def __init__(self, depth=1) -> None:
        super().__init__(depth=depth)

# ----------------------------

class Resnet152(UNet):
    def __init__(self, model=resnet152, depth=4) -> None:
        super().__init__(model, depth=depth)

class Resenet152Depth4(Resnet152):
    def __init__(self, depth=4) -> None:
        super().__init__(depth=depth)

class Resenet152Depth3(Resnet152):
    def __init__(self, depth=3) -> None:
        super().__init__(depth=depth)

class Resenet152Depth2(Resnet152):
    def __init__(self, depth=2) -> None:
        super().__init__(depth=depth)

class Resenet152Depth1(Resnet152):
    def __init__(self, depth=1) -> None:
        super().__init__(depth=depth)

import segmentation_models_pytorch as smp

def str2Model(string):
    if string == "UNetPPRes101":
        return lambda: smp.UnetPlusPlus(encoder_name='resnet101', encoder_weights='imagenet', classes=2)

    if string == "PSPRes101":
        return lambda: smp.PSPNet(encoder_name='resnet101', encoder_weights='imagenet', classes=2)

    if string == "UNetPP":
        return NestedUNet
    if string == "UNet":
        return U_Net
    if string == "TransUnet":
        return VitSeg
    if string == "AttnUnet":
        return AttU_Net

    if string == "Resenet18Depth4":
        return Resenet18Depth4
    if string == "Resenet18Depth3":
        return Resenet18Depth3
    if string == "Resenet18Depth2":
        return Resenet18Depth2
    if string == "Resenet18Depth1":
        return Resenet18Depth1
    
    if string == "Resenet34Depth4":
        return Resenet34Depth4
    if string == "Resenet34Depth3":
        return Resenet34Depth3
    if string == "Resenet34Depth2":
        return Resenet34Depth2
    if string == "Resenet34Depth1":
        return Resenet34Depth1
    
    if string == "Resenet50Depth4":
        return Resenet50Depth4
    if string == "Resenet50Depth3":
        return Resenet50Depth3
    if string == "Resenet50Depth2":
        return Resenet50Depth2
    if string == "Resenet50Depth1":
        return Resenet50Depth1

    if string == "Resenet101Depth4":
        return Resenet101Depth4
    if string == "Resenet101Depth3":
        return Resenet101Depth3
    if string == "Resenet101Depth2":
        return Resenet101Depth2
    if string == "Resenet101Depth1":
        return Resenet101Depth1
    
    if string == "Resenet152Depth4":
        return Resenet152Depth4
    if string == "Resenet152Depth3":
        return Resenet152Depth3
    if string == "Resenet152Depth2":
        return Resenet152Depth2
    if string == "Resenet152Depth1":
        return Resenet152Depth1
    
    raise ValueError("Invalid model name")

if __name__ == '__main__':
    model = Resenet18Depth1()
    model(torch.zeros(1, 3, 160, 160))
    model = Resenet18Depth2()
    model(torch.zeros(1, 3, 160, 160))
    model = Resenet18Depth3()
    model(torch.zeros(1, 3, 160, 160))
    model = Resenet18Depth4()
    model(torch.zeros(1, 3, 160, 160))

    model = Resenet34Depth1()
    model(torch.zeros(1, 3, 160, 160))
    model = Resenet34Depth2()
    model(torch.zeros(1, 3, 160, 160))
    model = Resenet34Depth3()
    model(torch.zeros(1, 3, 160, 160))
    model = Resenet34Depth4()
    model(torch.zeros(1, 3, 160, 160))

    model = Resenet50Depth1()
    model(torch.zeros(1, 3, 160, 160))
    model = Resenet50Depth2()
    model(torch.zeros(1, 3, 160, 160))
    model = Resenet50Depth3()
    model(torch.zeros(1, 3, 160, 160))
    model = Resenet50Depth4()
    model(torch.zeros(1, 3, 160, 160))

    model = Resenet101Depth1()
    model(torch.zeros(1, 3, 160, 160))
    model = Resenet101Depth2()
    model(torch.zeros(1, 3, 160, 160))
    model = Resenet101Depth3()
    model(torch.zeros(1, 3, 160, 160))
    model = Resenet101Depth4()
    model(torch.zeros(1, 3, 160, 160))

    model = Resenet152Depth1()
    model(torch.zeros(1, 3, 160, 160))
    model = Resenet152Depth2()
    model(torch.zeros(1, 3, 160, 160))
    model = Resenet152Depth3()
    model(torch.zeros(1, 3, 160, 160))
    model = Resenet152Depth4()
    model(torch.zeros(1, 3, 160, 160))