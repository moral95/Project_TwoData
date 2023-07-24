# .models/select_model.py

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, swin_b, Swin_B_Weights
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR

# class my_linear(nn.Module):

def model_selection(args, num_classes):
    if args == 'resnet50':
        model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V1)
        model.fc.out_features = num_classes

    elif args == 'resnet18':
        model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        model.fc.out_features = num_classes

    elif args== 'transformer':
        model = swin_b(weights = ResNet50_Weights.IMAGENET1K_V1)
        model.fc.out_features = num_classes
        
    return model

def criterion_selection(args):
    if args == 'CEL':
        criterion = CrossEntropyLoss()
    return criterion

def optim_selection(args, model, lr):
    if args == 'Adam':
        optim = Adam(model.parameters(), lr)
    return optim

def scheduler_selection(args, opt, step, gamma):
    if args == 'StepLR':
        scheduler = StepLR(opt, step, gamma)
    return scheduler



