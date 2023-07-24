

import torch
import torchvision
import argparse
import datetime
import logging
from torch.utils.tensorboard import SummaryWriter

from utils.dataset import *
from utils.plot import *
from utils.early_stopping import *
from models.select_model import *

from train import train, validate

########## Hyper Parameters #######
config = load_config('configs/configs.yaml')  # specify the path to your config file
writer = SummaryWriter()
early_stopping = EarlyStopping(patience=config['training']['patience'], 
                                verbose=True, 
                                path=config['paths']['model_save_path'])

###################################

def run():
    print(config)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") 
    print(f'We used the {device}')
    train_dataloader, valid_dataloader, test_dataloader = data_loader()
    model = model_selection(config['training']['model'], config['model']['num_classes'])
    model.to(device)
    criterion = criterion_selection(config['training']['criterion'])
    optimizer = optim_selection(config['training']['optim'], model, config['training']['learning_rate'])
    scheduler = scheduler_selection(config['training']['scheduler'], optimizer, config['training']['step'], config['training']['gamma'])
    
    ############ plot ############
    plot_dataset_distribution(config['paths']['train_annot'])
    plot_data_len()
    plot_image_sample(config['paths']['dataset_path'],config['paths']['train_annot'])
    plot_image_samplesx4(config['paths']['dataset_path'], config['paths']['train_annot'])


    best_loss, best_acc = 0, 0

    for epoch in range(config['training']['num_epochs']):
        train_loss, train_acc, train_precision, train_recall, train_f1, train_accuracy, train_confusion_matrix =train(model, device, train_dataloader, criterion, optimizer, config)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1, valid_accuracy, valid_confusion_matrix = validate(model, device, valid_dataloader, criterion)
        scheduler.step()

        if valid_acc > best_acc or (valid_acc == best_acc and valid_loss < best_loss):
            best_acc, best_loss = valid_acc, valid_loss
        print(f"""
              Epoch: {epoch+1}/{config['training']['num_epochs']}.. 
              Training Loss: {train_loss:.4f}.. Training Accuracy: {train_acc:.2f}%.. 
              Training Precision: {train_precision:.2f}..
              Training Recall: {train_recall:.2f}..
              Training F1 Score: {train_f1:.2f}..
              Training Accuracy: {train_accuracy:.2f}..
              Train Confusion Matrix:\n, {train_confusion_matrix}
            """)
        print(f"""
              Epoch: {epoch+1}/{config['training']['num_epochs']}.. 
              Validation Loss: {valid_loss:.4f}.. Validation Accuracy: {valid_acc:.2f}%.. 
              Validation Precision: {valid_precision:.2f}..
              Validation Recall: {valid_recall:.2f}..
              Validation F1 Score: {valid_f1:.2f}..
              Validation Accuracy: {valid_accuracy:.2f}..
              Validation Confusion Matrix:\n, {valid_confusion_matrix}              

              Best_Validation Loss: {best_loss:.4f}.. Best_Validation Accuracy: {best_acc:.2f}%.. 
              """)
        
        ########### early stopping ###########
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    ############ writer ##########
    writer.add_scalar('Train/F1 Score', train_f1, config['training']['num_epochs'])
    writer.add_scalar('Valid/F1 Score', valid_f1, config['training']['num_epochs'])
    writer.add_scalar('Train/Loss', train_loss/len(train_dataloader), config['training']['num_epochs'])
    writer.add_scalar('Train/Accuracy', train_accuracy, config['training']['num_epochs'])
    writer.add_scalar('Train/Precision', train_precision, config['training']['num_epochs'])
    writer.add_scalar('Train/Recall', train_recall, config['training']['num_epochs'])
    writer.add_scalar('Train/F1 Score', train_f1, config['training']['num_epochs'])

    writer.add_scalar('Validation/Loss', valid_loss/len(valid_dataloader), config['training']['num_epochs'])
    writer.add_scalar('Validation/Accuracy', valid_accuracy, config['training']['num_epochs'])
    writer.add_scalar('Validation/Precision', valid_precision, config['training']['num_epochs'])
    writer.add_scalar('Validation/Recall', valid_recall, config['training']['num_epochs'])
    writer.add_scalar('Validation/F1 Score', valid_f1, config['training']['num_epochs'])        
    writer.close()

    # PATH = './save/butter_%f_%d.pth' % (config['training']['learning_rate'], 5657)
    torch.save(model.state_dict(), config['paths']['model_save_path'])


if __name__ == "__main__":
    run()
