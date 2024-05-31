from model_training.trainer import Trainer
from model_training.dataset import CAGDataset
from torch.utils.data import DataLoader
import math
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='')
args = parser.parse_args()
config = 'config/' + args.config + '.yaml'

if config == "":
    raise ValueError("Please specify the config file path.")

with open(config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

background_img = True
if 'background_img' in config and config['background_img'] == False:
    background_img = False

dataset = CAGDataset(background_img=background_img)
# shuffle make big difference
train_loader = DataLoader(dataset, config['batch_size'], num_workers=config['num_workers'], shuffle=True)
model = Trainer(config=config)

total_iter = 0

if config['load_checkpoint'] != None:
    total_iter = model.load_checkpoint(config['load_checkpoint'])
    config['load_checkpoint'] = None
    print('Previously trained model loaded!')

model.train()


total_epoch = math.ceil(config['iterations']/len(train_loader))
current_epoch = total_iter // len(train_loader)
current_epoch = total_iter // len(train_loader)
print(f'We approximately use {total_epoch} epochs.')

try:
    while total_iter < config['iterations']:
        current_epoch += 1
        print(f'Current epoch: {current_epoch}')

        # Train loop
        for data in train_loader:

            model.do_pass(data, total_iter)
            total_iter += 1

            if total_iter >= config['iterations']:
                break
finally:
    if not config['debug'] and model.logger is not None and total_iter>50:
        model.save_checkpoint(total_iter)
