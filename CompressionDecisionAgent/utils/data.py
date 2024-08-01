import h5py
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

class CompressionDataset(Dataset):
    def __init__(self, file_name, transform):
        self.data = self._load_h5_file(file_name)
        self.transform = transform

    def _load_h5_file(self, file_name):
        return h5py.File(file_name)
    
    def __len__(self):
        return len(self.data['Images'])

    def _compute_dice_score(self, mask1, mask2, eps=1e-6):
        mask1 = mask1.numpy()
        mask2 = mask2.numpy()
        
        intersection = np.logical_and(mask1, mask2) + eps
        
        return 2 * np.sum(intersection) / (np.sum(mask1) + np.sum(mask2) + eps)
        
    def __getitem__(self, idx):
        image = self.data['Images'][idx]
        HR_mask = self.data['HR_masks'][idx]
        LR_mask = self.data['LR-x4_masks'][idx]

        transformed = self.transform(
            image=image, mask=HR_mask, mask1=LR_mask
        )
        
        result_dict = {
            "orig_image": image,
            "image": transformed['image'],
            "dice": torch.tensor(self._compute_dice_score(transformed['mask'], transformed['mask1']))
        }
        
        return result_dict

class InferenceDataset(Dataset):
    def __init__(self, HR_flist, transforms=None):
        super().__init__()
        self.HR_flist = HR_flist
        self.transforms = transforms
    
    def _read_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return np.array(image)
    
    def __len__(self):
        return len(self.HR_flist)
    
    def __getitem__(self, idx):
        fname = self.HR_flist[idx]
        image = self._read_image(fname)
        image = self.transforms(image=image)['image']
        
        return fname, image

def define_augmentations():
    train_transform = A.Compose(
        [
            A.RandomCrop(256, 256, p=0.9),
            
            A.OneOf(
                [
                    A.MotionBlur(),
                    A.MedianBlur(blur_limit=3),
                    A.Blur(blur_limit=3),
                ], p=0.2),
        
            A.OneOf([
                A.ChannelShuffle(),
                A.ColorJitter(),
                A.HueSaturationValue(),
            ], p=0.5),
            
            A.Resize(224, 224, p=1),
            A.Normalize(p=1),
            ToTensorV2(p=1),
        ],
        
        additional_targets={
            'mask1': 'mask'
        }
    )

    valid_transform = A.Compose(
        [
            A.Resize(224, 224, p=1),
            A.Normalize(p=1),
            ToTensorV2(p=1),
        ],
        
        additional_targets={
            'mask1': 'mask'
        }
    )

    return train_transform, valid_transform

def define_datasets_and_dataloaders(train_transform, valid_transform,
                                    batch_size=64, num_workers=16):
    train_dataset = CompressionDataset("./data/DynamicCompressionDataset_train.h5", train_transform)
    valid_dataset = CompressionDataset("./data/DynamicCompressionDataset_valid.h5", valid_transform)
    test_dataset = CompressionDataset("./data/DynamicCompressionDataset_test.h5", valid_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=True, persistent_workers=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=True)
    
    return train_dataloader, valid_dataloader, test_dataloader

def get_inference_image_flist(path):
    import glob
    flist = sorted(glob.glob(path))
    
    return flist

def define_inference_dataset_and_dataloader(inference_flist, 
                                            batch_size=64, 
                                            num_workers=16):
    
    _, valid_transform = define_augmentations()
    inference_dataset = InferenceDataset(inference_flist, valid_transform)
    inference_dataloader = DataLoader(inference_dataset, 
                                      shuffle=False,
                                      batch_size=batch_size,
                                      num_workers=num_workers,
                                      pin_memory=True)
    
    return inference_dataloader