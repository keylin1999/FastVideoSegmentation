from model_training.network import str2Model
from model_training.loss import LossComputer
from model_training.logger import TensorboardLogger, Integrator
import random
import time
import torch
import math
import os
import yaml

class Trainer:
    def __init__(self, config) -> None:
        self.config = config
        self.model = str2Model(config["model"])().cuda()

        self.save_path = 'output/' + config['exp_id']
        if not os.path.exists(self.save_path):
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        else: 
            raise ValueError(f"Directory {self.save_path} already exists.")
        logger = TensorboardLogger(config['exp_id'], "git_info", root_path='output/')
        self.logger = logger
        logger.log_string('hyperpara', str(config))
        self.integrator = Integrator(self.logger, distributed=False)


        # loss
        self.loss_computer = LossComputer(config)
        self.loss_computer2 = LossComputer(config)
        self.last_time = time.time()
        
        # training hyper-parameters
        self.train()
        self.optimizer = torch.optim.AdamW(filter(
            lambda p: p.requires_grad, self.model.parameters()), lr=config['lr'], weight_decay=config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, config['steps'], config['gamma'])

        self.log_text_interval = config['log_text_interval']
        self.log_image_interval = config['log_image_interval']
        self.save_checkpoint_interval = config['save_checkpoint_interval']
        self.save_checkpoint_its = config['save_checkpoint_its']
        if config['debug']:
            self.log_text_interval = self.log_image_interval = 1
        

    def do_pass(self, data, it=0):
        image, target = data
        image = image.cuda()
        target = target.cuda()
        out = {}
        pred_gt, pred_logit = self.model(image)

        out[f'mask'] = pred_gt
        out[f'logit'] = pred_logit
        out['target'] = target

        losses = self.loss_computer.compute(out, it)
        if 'random_crop' in self.config and self.config['random_crop']:
            size = random.randint(128, 320)
            size = math.ceil(size/16)*16
            start_x = random.randint(0, 384-size)
            start_y = random.randint(0, 384-size)
            image2 = image[:,:,start_x:start_x+size,start_y:start_y+size]
            target2 = target[:,start_x:start_x+size,start_y:start_y+size]
            pred_gt2, pred_logit2 = self.model(image2)
            out2 = {}
            out2[f'mask'] = pred_gt2
            out2[f'logit'] = pred_logit2
            out2['target'] = target2
            losses2 = self.loss_computer2.compute(out2, it)
            for key in losses2:
                losses[key] += losses2[key]

        if self._do_log:
            self.integrator.add_dict(losses)
            if it % self.log_image_interval == 0 and it != 0:
                label = pred_gt[0].argmax(0).float()
                self.logger.log_img('train/pairs', torch.cat([image[0].cpu(),target[0].float().repeat((3,1,1)).cpu(), label.repeat(3,1,1).cpu()], dim=2).numpy(), it)

        if self._is_train:
            if (it) % self.log_text_interval == 0 and it+1 != 0:
                self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.log_text_interval, it)
                self.last_time = time.time()
                self.integrator.finalize('train', it)
                self.integrator.reset_except_hooks()

            if self.save_checkpoint_interval is not None:
                if it % self.save_checkpoint_interval == 0 and it != 0:
                    self.save_checkpoint(it)
            else:
                if it in self.save_checkpoint_its:
                    self.save_checkpoint(it)
        
        self.optimizer.zero_grad(set_to_none=True)
        losses['total_loss'].backward()
        self.optimizer.step()
        self.scheduler.step()

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = f'{self.save_path}/checkpoint_{it}.pth'
        checkpoint = { 
            'it': it,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'model_type': self.config["model"]
        }
        torch.save(checkpoint, checkpoint_path)

        json_path = f'{self.save_path}/config.yaml'
        # check if exist
        if not os.path.exists(json_path):
            with open(json_path, 'w') as f:
                yaml.dump(self.config, f)
        print(f'Checkpoint saved to {checkpoint_path}.')

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        self.model.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Network weights, optimizer states, and scheduler states loaded.')

        return it

    def train(self):
        self._is_train = True
        self._do_log = True
        self.model.eval()
    
    def val(self):
        self._is_train = False
        self._do_log = True
        self.model.eval()

    def test(self):
        self._is_train = False
        self._do_log = False
        self.model.eval()


if __name__ == "__main__":
    with open("/home/charlieyao/XMem_total/thesis/config/resnet18.yaml", "r") as f:
        print(yaml.load(f, Loader=yaml.FullLoader))