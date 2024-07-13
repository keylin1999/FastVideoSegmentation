import argparse
from model_training.dataset import CAGTestDataset
from model_training.network import str2Model, Resenet18Depth3
from torch.utils.data import DataLoader
from PIL import Image
from metrics import Metrics
from torchvision import transforms
import os
import torch
from torchprune.util.net import NetHandle
from torchprune.method.alds import ALDSNet
from torchprune.method.pca import PCANet
from torchprune.method.pfp import PFPNet
from torchprune.method.sipp import SiPPNet
from torchprune.method.thres_filter import FilterThresNet

def method2net(method):
    if method == 'alds':
        return ALDSNet
    elif method == 'pca':
        return PCANet
    elif method == 'pfp':
        return PFPNet
    elif method == 'sipp':
        return SiPPNet
    elif method == 'thres':
        return FilterThresNet
    else:
        raise ValueError('Unknown method')
    
def filename2method(file_name):
    fa = file_name[32]
    fb = file_name[33]
    if fa == 'A':
        return 'alds'
    elif fa == 'P':
        if fb == 'C':
            return 'pca'
        elif fb == 'F':
            return 'pfp'
    elif fa == 'S':
        return 'sipp'
    elif fa == 'F':
        return 'thres'
    
work_dir = '/home/charlieyao/XMem_total/torchprune/data/results/cagdataset/Resenet18Depth3/2024_01_30_10_14_31_328586/retrained_networks/'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='')
args = parser.parse_args()

model = torch.load(work_dir + args.model)
model2 = NetHandle(Resenet18Depth3())
model2 = method2net(filename2method(args.model))(model2, [], lambda x, y: x)

model = model['net']
model2.load_state_dict(model, strict=False)
model = model2.cuda().eval()

testDataset = CAGTestDataset()
testLoader = DataLoader(testDataset, 1, num_workers=1)
metric = Metrics()

for data in testLoader:
    image, target = data
    image = image.cuda()
    target = target.cuda()
    gt, logit = model(image)
    gt = gt.argmax(1)[0].cpu().numpy()
    target = target[0].cpu().numpy()
    metric.update_np(gt, target)

metric.print()
print(metric.count)

