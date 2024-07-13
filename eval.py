import argparse
from model_training.dataset import CAGTestDataset
from model_training.dca import DCA1Test
from model_training.network import str2Model
from torch.utils.data import DataLoader
from PIL import Image
from metrics import Metrics
from torchvision import transforms
import os
import torch

device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='')
parser.add_argument("--dset", type=str, default='CAG')
args = parser.parse_args()

model = torch.load(args.model, map_location=device)
mtype = model['model_type']
model2 = str2Model(model['model_type'])()
model = model['network']
model2.load_state_dict(model)
model = model2.to(device).eval()

testDataset = CAGTestDataset(background_img=False)
if args.dset != 'CAG':
    testDataset = DCA1Test()
testLoader = DataLoader(testDataset, 1, num_workers=1)
metric = Metrics()

for data in testLoader:
    image, target = data
    image = image.to(device)
    target = target.to(device)
    if mtype[0] != "R":
        logit = model(image)
        gt = logit.argmax(1)[0].cpu().numpy()
    else:
        gt, logit = model(image)
        gt = gt.argmax(1)[0].cpu().numpy()
    target = target[0].cpu().numpy()
    metric.update_np(gt, target)

metric.print()
print(metric.count)

