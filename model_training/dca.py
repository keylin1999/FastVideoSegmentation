from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from PIL import Image
import torch
import numpy as np
import os
import cv2 as cv
import random

def sort_file(files):
    temp = list(map(lambda x: [len(x), x], files))
    temp.sort()
    return list(map(lambda x: x[1], temp))

def get_files(file_dir):
    files = os.listdir(file_dir)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    return files


class DCA1(Dataset):
    def __init__(self, train=True, noAug=False, file_dir='/home/charlieyao/XMem_total/training_dataset/Database_134_Angiograms/'):
        random.seed(0)
        arange = list(range(1, 135))
        random.shuffle(arange)
        train_files = arange[:100]
        test_files = arange[100:]

        files = train_files if train else test_files
        files.sort()

        self.images = [f"{file_dir}{file}.pgm" for file in files]
        self.targets = [f"{file_dir}{file}_gt.pgm" for file in files]
        self.image_info = [["DCA1", f"{file}.pgm"] for file in files]
        self.noAug = noAug
        self.train = train

    def _image_transform(self, image):
        img = Image.open(image).convert('RGB')
        img = np.array(img)
        img2 = img.copy()
        for _ in range(0):
            dst = cv.fastNlMeansDenoisingColored(img,None,3,5,5,5)
            img = dst
        img = cv.addWeighted(img, 0.4, img2, 0.6, 0)
        img = Image.fromarray(img)
        img = img.resize((512, 512))
        img = transforms.ToTensor()(img)
        return img
    
    def _target_transform(self, target):
        target = Image.open(target)
        target = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST)(target)
        target = transforms.ToTensor()(target)
        target.squeeze_(0)
        target = torch.as_tensor(target, dtype=torch.int64)
        return target
    
    def _image_transform_aug(self, image):
        imgsize = 512 if self.noAug else 384
        im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.2, 0.4, 0.4, 0)
        ])
        im_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, shear=0, interpolation=transforms.InterpolationMode.BICUBIC, fill=(0, 0, 0)),
            transforms.RandomResizedCrop((imgsize, imgsize), scale=(0.36,1.00), interpolation=transforms.InterpolationMode.BILINEAR),
        ])


        img = Image.open(image).convert('RGB')
        self.reseed()
        img = im_lone_transform(img)
        self.reseed()
        img = im_dual_transform(img)
        img = transforms.ToTensor()(img)

        return img        
        

    def _target_transform_aug(self, target):
        imgsize = 512 if self.noAug else 384
        gt_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, shear=0, interpolation=transforms.InterpolationMode.NEAREST, fill=0),
            transforms.RandomResizedCrop((imgsize, imgsize), scale=(0.36,1.00), interpolation=transforms.InterpolationMode.NEAREST),
        ])

        target = Image.open(target)
        self.reseed()
        target = gt_dual_transform(target)
        target = transforms.ToTensor()(target)
        target.squeeze_(0)
        target = torch.as_tensor(target, dtype=torch.int64)
        return target

    def reseed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
    def set_seed(self):
        seed = np.random.randint(2147483647)
        self.seed = seed


    def __getitem__(self, index):
        img = self.images[index]
        target = self.targets[index]
        self.set_seed()

        if self.train:
            img = self._image_transform_aug(img)
        else:
            img = self._image_transform(img)
        if self.train:
            target = self._target_transform_aug(target)
        else:
            target = self._target_transform(target)

        return img, target
    
    def __len__(self):
        return len(self.images)
    
class DCA1Test(DCA1):
    def __init__(self, noAug=False, file_dir='/home/charlieyao/XMem_total/training_dataset/Database_134_Angiograms/'):
        super().__init__(False, noAug, file_dir)


if __name__ == '__main__':
    dataset = DCA1()
    print(len(dataset))
    print(dataset[0])
    