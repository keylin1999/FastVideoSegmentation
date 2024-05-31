# import datasets class
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from PIL import Image
import torch
import numpy as np
import os
import cv2 as cv

last_file = {
    'CAU7': 38,
    'CRA17': 61,
    'CRA27': 38,
    'CAU31': 49,
    'CRA31': 53,
    'CRA34': 50,
    'CAU26': 43,
    'CRA0': 52,
    'CRA9': 55,
    'CRA23': 56
}

def sort_file(files):
    temp = list(map(lambda x: [len(x), x], files))
    temp.sort()
    return list(map(lambda x: x[1], temp))

def get_files(file_dir):
    files = os.listdir(file_dir)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    return files

class CAGDataset(Dataset):
    def __init__(
        self,
        train=True,
        background_img=True,
        file_dir="/home/charlieyao/XMem_total/training_dataset/static_type0_first_blank_copy"
    ):
        self.download = False
        self.background_img = background_img
        self.train = train
        self.seed = 0
        image_dir = os.path.join(file_dir, "JPEGImages")
        target_dir = os.path.join(file_dir, "Annotations")

        self.images = []
        self.targets = []
        self.image_info = []

        # get list of images and targets
        videos = get_files(image_dir)
        for video in videos:
            video_dir = os.path.join(image_dir, video)
            video_targets = os.path.join(target_dir, video)
            images = get_files(video_dir)
            images = sort_file(images)
            if video in last_file:
                images = images[:last_file[video] - 10 + 1]
            first_img = images[0]
            for image in images[1:]:
                image_path = os.path.join(video_dir, image)
                target_path = None
                if os.path.exists(os.path.join(video_targets, image)):
                    target_path = os.path.join(video_targets, image)
                elif os.path.exists(os.path.join(video_targets, image.replace(".jpg", ".png"))):
                    target_path = os.path.join(video_targets, image.replace(".jpg", ".png"))
                else:
                    raise Exception("Target file does not exist")
                self.images.append([image_path, os.path.join(video_dir, first_img)])
                self.targets.append(target_path)
                self.image_info.append([video, image])
    
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
        im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.2, 0.4, 0.4, 0)
        ])
        im_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, shear=0, interpolation=transforms.InterpolationMode.BICUBIC, fill=(0, 0, 0)),
            transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=transforms.InterpolationMode.BILINEAR),
        ])


        img = Image.open(image).convert('RGB')
        self.reseed()
        img = im_lone_transform(img)
        self.reseed()
        img = im_dual_transform(img)
        img = transforms.ToTensor()(img)

        return img        
        

    def _target_transform_aug(self, target):
        gt_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, shear=0, interpolation=transforms.InterpolationMode.NEAREST, fill=0),
            transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=transforms.InterpolationMode.NEAREST),
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
        img, first_img = self.images[index]
        target = self.targets[index]
        self.set_seed()

        if self.train:
            img = self._image_transform_aug(img)
            first_img = self._image_transform_aug(first_img)
        else:
            img = self._image_transform(img)
            first_img = self._image_transform(first_img)
        if self.background_img:
            img[0, :, :] = first_img[0, :, :]
        if self.train:
            target = self._target_transform_aug(target)
        else:
            target = self._target_transform(target)

        return img, target
    
    def __len__(self):
        return len(self.images)

class CAGTestDataset(CAGDataset):
    def __init__(
        self,
        train=False,
        background_img=True
    ) -> None:
        
        self.file_dir = "/home/charlieyao/XMem_total/eval_dataset/bounding_box"
        self.download = False

        super().__init__(train, file_dir=self.file_dir, background_img=background_img)
    
if __name__ == '__main__':
    dataset = CAGDataset()
    print(len(dataset))
    dataset[0]