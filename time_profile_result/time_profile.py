import argparse
from model_training.dataset import CAGTestDataset
from model_training.network import str2Model
import pickle
import torch
from torchvision import transforms
import time

class MockEvent:
    def __init__(self):
        pass
    def record(self):
        self.time = time.time()
    def elapsed_time(self, end):
        return (end.time - self.time) * 1000

args = argparse.ArgumentParser()
args.add_argument('-b', '--box_path', type=str, default='')
args.add_argument('-d', '--device', type=str, default='mps:0')
args = args.parse_args()

print(args.box_path)

# because the bug in video segmentation, even if stage 2 is not performed, it still add some boxes
# if the stage 2 is performed than it has no effect
#192_8_128_0.9_None_None
first_file = {
    'CAU7': list(range(6)) + [7],
    'CRA17': [0, 1, 2, 4, 6, 9],
    'CRA27': [0, 1, 2, 6, 7],
    'CAU31': list(range(6)),
    'CRA31': list(range(11)),
    'CRA34': list(range(4)),
    'CAU26': list(range(8)),
    'CRA9': [0, 1, 2, 6, 7, 8, 9],
    'CRA23': list(range(11))
}



last_file = {
    'CAU7': 38,
    'CRA17': 61,
    'CRA27': 38,
    'CAU31': 49,
    'CRA31': 53,
    'CRA34': 50,
    'CAU26': 43,
    'CRA9': 55,
    'CRA23': 56
}

start = torch.mps.Event(enable_timing=True)
end = torch.mps.Event(enable_timing=True)
if args.device == 'cpu':
    start = MockEvent()
    end = MockEvent()

dataset = CAGTestDataset()
boxes = None
with open(args.box_path, 'rb') as f:
    boxes = pickle.load(f)


model = torch.load('output/checkpoint_50000.pth', map_location=torch.device('cpu'))
model2 = str2Model(model['model_type'])()
model = model['network']
model2.load_state_dict(model)
model = model2.eval()
model = model.to(args.device)

stage1_time = 0
stage2_time = 0

def measure_time(nimg, args):
    for _ in range(5):
        model(torch.randn_like(nimg))
    torch.mps.synchronize()
    start.record()
    model(nimg)
    end.record()
    if args.device == 'mps:0':
        torch.mps.synchronize()
    gpu_time = start.elapsed_time(end)
    return gpu_time

for idx, (img, _) in enumerate(dataset):
    video, file_name = dataset.image_info[idx]
    file_name = file_name.split('.')[0]
    file_name = int(file_name)
    l = len(boxes[video])
    fidx = last_file[video] - l + 1
    file_idx = file_name - fidx
    box_info = boxes[video][file_idx]

    for box in box_info:
        left, right, top, bottom = box
        
        elapse_time = None
        stage = "stage1"
        nimg = None
        if left == right and right == top and top == bottom:
            img = img.to('cpu')
            nimg = transforms.Resize((left, left))(img)
            nimg = nimg.to(args.device)
            nimg = nimg.unsqueeze(0)
            elapse_time = measure_time(nimg, args)
            stage1_time += elapse_time
            print(idx, "{:.4f}".format(elapse_time), stage, video)
        else:
            # because of bug, adaptive will add at least one more box
            if (file_idx not in first_file[video]): # or True: # for plain usage
                img = img.to(args.device)
                nimg = img[:, top:bottom, left:right]
                nimg = nimg.unsqueeze(0)
                elapse_time = measure_time(nimg, args)
                stage2_time += elapse_time
                stage = "stage2"
                print(idx, "{:.4f}".format(elapse_time), stage, video)
        
print(f"stage1_time: {stage1_time}, stage2_time: {stage2_time}", "\n")


        