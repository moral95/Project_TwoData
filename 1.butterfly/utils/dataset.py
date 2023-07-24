# ./utils/dataset.py
import os
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils.config import load_config
from sklearn.model_selection import train_test_split
from utils.transform import transform_selection
import warnings
warnings.filterwarnings("ignore")

class ButterflyDataset(Dataset):
    def __init__(self, annotations_file, dataset_path, mode, seed, transform=None, target_transform=None):
        self.config = load_config('configs/configs.yaml')  # moved here
        self.img = pd.read_csv(annotations_file)
        self.mode = mode
        train_dataset, val_dataset = train_test_split(self.img, test_size=0.3, random_state = seed)
        if mode == 'train':
            self.img_mod = train_dataset
        elif mode == 'valid':
            self.img_mod = val_dataset
        elif mode == 'test':
            self.img_mod = self.img
        else:
            print(f'{mode} is NOT suitable Mode')

        self.transform = transform
        self.target_transform = target_transform
        self.dataset_path = dataset_path
        if self.mode == 'train' or 'valid':
            self.dataset_path = self.dataset_path + 'train'
        else:
            self.dataset_path = self.dataset_path + self.img
               
    def __len__(self):
        return len(self.img_mod)
        
    def __getitem__(self, idx):
        # 이미지 경로 설정 및 이미지 반환
        img_path = self.img_mod.iloc[idx, 0]
        img_path = os.path.join(self.dataset_path, img_path)
        image = self.load_image(img_path)
        
        # 이미지 transform 설정
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            image = self.target_transform(image)
        
        # 이미지 label 설정
        raw_label = self.img_mod.iloc[idx, 1]
        label_set = set(raw_label)
        dic_label = {}
        for idx, label in enumerate(label_set):
            dic_label[idx] = label
        
        for k, v in dic_label.items():
            if v == raw_label[idx]:
                label = k
        return image, label

    def load_image(self, path):
        with Image.open(path) as img:
            img.load()  # This forces the image file to be read into memory
            return img  # Return the PIL image directly

def seed_everything(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def data_loader():
    transform, target_transform = transform_selection()
    # transform, target_transform, transfrom_album = transform_selection()
    config = load_config('configs/configs.yaml')  # moved here

    seed = seed_everything(config['training']['seed'])

    train_data  = ButterflyDataset(config['paths']['train_annot'], config['paths']['dataset_path'], mode='train',seed=seed, transform=transform)
    train_loader = DataLoader(train_data, batch_size = config['training']['batch_size'], shuffle = True)


    valid_data  = ButterflyDataset(config['paths']['train_annot'], config['paths']['dataset_path'], mode='valid', seed=seed,transform=target_transform)
    valid_loader = DataLoader(valid_data, batch_size = config['training']['batch_size'], shuffle = False)

    test_data  = ButterflyDataset(config['paths']['test_annot'],config['paths']['dataset_path'], mode='test',seed=seed, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size = config['training']['batch_size'], shuffle = False)

    return train_loader, valid_loader, test_loader