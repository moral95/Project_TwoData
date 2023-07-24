# ./utils/transforms.py

import torch.nn as nn
# import albumentations
from torchvision import transforms

def transform_selection():

    transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(15),
                    transforms.CenterCrop(64),
                    transforms.GaussianBlur(5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])                    

    target_transform = transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                        ])
    
    ################### 외부 라이브러리 - albumentations ################

    # transform_album = albumentations.Compose([
    #                     albumentations.Resize(224),
    #                     albumentations.OneOf([
    #                     albumentations.RandomBrightness(p = 0.5),
    #                     albumentations.RandomContrast(p = 0.5),
    #                     albumentations.ColorJitter(p = 0.5),
    #                     ]),
    #                     albumentations.OneOf([
    #                     albumentations.RandomRain(p = 0.5),
    #                     albumentations.RandomFog(p = 0.5),
    #                     albumentations.RandomSnow(p = 0.5),
    #                     ]),
    #                     albumentations.Blur,
    #                     albumentations.CenterCrop(168),
    #                     albumentations.GaussNoise,
    #                     albumentations.pytorch.transforms.ToTensor,
    #                     albumentations.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),

    # ])


    return transform, target_transform
    # return transform, target_transform, transform_album


# transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(15),
#         transforms.CenterCrop(64),
#         transforms.GaussianBlur(5),
#         transforms.ToTensor(),
#     ])                    