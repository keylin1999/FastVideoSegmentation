from model_training.vit_seg_modeling import VisionTransformer as Vit_Seg
from model_training.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.patches.grid = (int(512 / 16), int(512 / 16))

def VitSeg():
    return Vit_Seg(config=config_vit, img_size=512, num_classes=2) 

# model = Vit_Seg(config=config_vit, img_size=384, num_classes=1)
# import torch
# print(model(torch.randn(1, 3, 384, 384)))