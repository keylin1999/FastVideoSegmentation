import argparse
from model_training.dataset import CAGTestDataset
from model_training.network import str2Model
from torch.utils.data import DataLoader
from PIL import Image
from metrics import Metrics
from torchvision import transforms
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='')
args = parser.parse_args()

model = torch.load(args.model)
model2 = str2Model(model['model_type'])()
model = model['network']
model2.load_state_dict(model)
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

