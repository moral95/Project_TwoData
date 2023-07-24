
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from utils.early_stopping import EarlyStopping
from utils.dataset import McDataset
from utils.config import load_config
from utils.plot import *
# from torch import binary_cross_entropy_with_logits # 해당 데이터셋은 binary가 아니다. 다른 걸로 보도록
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmarks = True  

config = load_config('configs/configs.yaml')
print(config)
seed = seed_everything(config['training']['seed'])
device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()

########## Hyper Parameters #######
lr = config['training']['learning_rate']
num_epochs = config['training']['num_epochs']
max_len = 512
annotation_file = config['paths']['annotation_path']

########## model Architecture ########
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
num_features = model.classifier.in_features
optimizer = torch.optim.AdamW(model.parameters(), lr= lr)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 5, gamma = 0.1)
early_stopping = EarlyStopping(patience=config['training']['patience'], 
                                verbose=True, 
                                path=config['paths']['model_save_path'])
########################################
class Bert(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.classifier = nn.Softmax()
        
    def forward(self, x):
        x= self.model(x)
        output = self.classifier(x)

        return output

model.classifier = nn.Sequential(
    nn.Linear(num_features, num_features//2),
    nn.ReLU(),
    nn.Linear(num_features//2, 5),
    nn.Softmax(),
    )
    
model.to(device)


######### DATA SET ########
train_dataset = McDataset(annotation_file, tokenizer, mode='train', seed= seed, max_len=max_len)
train_loader = DataLoader(train_dataset, batch_size= config['training']['batch_size'], shuffle = True)
print(f'train_dataset : {len(train_dataset)}')

val_dataset = McDataset(annotation_file, tokenizer, mode='val', seed= seed, max_len=max_len)
val_loader = DataLoader(val_dataset, config['training']['batch_size'], shuffle = False)
print(f'val_dataset : {len(val_dataset)}')

######### train, val def ######

########## Train #########
def train(model, criterion, optimizer, dataloader, device):
    model.train()
    optimizer.zero_grad()
    running_loss, total_predictions, correct_predictions = 0, 0, 0
    all_predictions = []
    all_labels = []
    for ids, masks, targets  in tqdm(dataloader):

        input_ids = ids.to(device)
        attention_mask = masks.to(device)
        labels = targets.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        outputs_max = torch.max(outputs.logits, 1)[0]
        loss = criterion(outputs_max, labels).to(device)
        loss.backward()

        # Gradient clipping
        if config['training']['gradient_clipping']:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5, norm_type=1.0)
        optimizer.step()
        running_loss += loss.item() 
 
        predict = torch.argmax(outputs.logits, 1)
        total_predictions +=  labels.size(0)
        correct_predictions += (predict == labels).sum().item()
        all_predictions.append(predict.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    train_precision = precision_score(all_labels, all_predictions, average='weighted')
    train_recall = recall_score(all_labels, all_predictions, average='weighted')
    train_f1 = f1_score(all_labels, all_predictions, average='weighted')
    train_accuracy = accuracy_score(all_labels, all_predictions)
    train_confusion_matrix = confusion_matrix(all_labels, all_predictions)

    print(f'len_dataloader : {len(dataloader)}')
    print(f'train_acc : {correct_predictions / total_predictions * 100}')
    print(f'train_loss : {running_loss / len(dataloader)}')
    print(f'train_loss_dataset : {running_loss / len(dataloader.dataset)}')

    return train_precision, train_recall, train_f1, train_accuracy, train_confusion_matrix, running_loss

########## Validation #########
def val(model, criterion, dataloader, device):
    model.eval()

    running_valid_loss, total_valid_predictions, correct_valid_predictions = 0, 0, 0
    all_valid_labels, all_valid_predictions = [], []

    with torch.no_grad():
        for ids, mask, label in tqdm(dataloader):
            input = ids.to(device)
            mask = mask.to(device)
            label = label.to(device)

            output = model(input, attention_mask = mask)
            outputs_max = torch.max(output.logits, 1)[0]
           
            loss = criterion(outputs_max, label).to(device)

            running_valid_loss += loss.item()
            total_valid_predictions += ids.size(0)

            predict = torch.argmax(output.logits, 1)
            
            correct_valid_predictions = (predict == label).sum().item()
            all_valid_predictions.append(predict.detach().cpu().numpy())
            all_valid_labels.append(label.detach().cpu().numpy())            

    all_predictions = np.concatenate(all_valid_predictions)
    all_labels = np.concatenate(all_valid_labels)
    valid_precision = precision_score(all_labels, all_predictions, average='weighted')
    valid_recall = recall_score(all_labels, all_predictions, average='weighted')
    valid_f1 = f1_score(all_labels, all_predictions, average='weighted')
    valid_accuracy = accuracy_score(all_labels, all_predictions)
    valid_confusion_matrix = confusion_matrix(all_labels, all_predictions)

    print(f'valid_acc : {correct_valid_predictions / total_valid_predictions *100}')
    print(f'valid_loss : {running_valid_loss / len(dataloader)}')
    print(f'valid_loss_dataset : {running_valid_loss / len(dataloader.dataset)}')

    return valid_precision, valid_recall, valid_f1, valid_accuracy, valid_confusion_matrix, running_valid_loss

########## Training #########
def training():
    config = load_config("configs/configs.yaml")
    
    print(f'We use the {device}')
    plot_dataset_distribution()
    plot_data_length()
    plot_split_data_length()

    best_val_loss = 0
    for epochs in range(num_epochs):
        train_precision, train_recall, train_f1, train_accuracy, train_confusion_matrix, epoch_loss = train(model, criterion, optimizer, train_loader, device)
        valid_precision, valid_recall, valid_f1, valid_accuracy, valid_confusion_matrix, val_loss = val(model, criterion, val_loader, device)
        scheduler.step()

        print('Train Confusion Matrix:\n', train_confusion_matrix)
        print(f'''
            Epoch: {epochs}, 
            Loss: {epoch_loss/len(train_loader):.4f}, 
            Precision: {train_precision:.4f}, 
            Recall: {train_recall:.4f}, 
            F1 Score: {train_f1:.4f}, 
            Accuracy: {train_accuracy:.4f}
            ''')
        
        print('Valid Confusion Matrix:\n', valid_confusion_matrix)
        print(f'''
            Epoch: {epochs}, 
            Loss: {val_loss/len(train_loader):.4f}, 
            Precision: {valid_precision:.4f}, 
            Recall: {valid_recall:.4f}, 
            F1 Score: {valid_f1:.4f}, 
            Accuracy: {valid_accuracy:.4f}
            ''')

        writer.add_scalar('Train/Loss', epoch_loss/len(train_loader), epochs)
        writer.add_scalar('Train/Accuracy', train_accuracy, epochs)
        writer.add_scalar('Train/Precision', train_precision, epochs)
        writer.add_scalar('Train/Recall', train_recall, epochs)
        writer.add_scalar('Train/F1 Score', train_f1, epochs)

        writer.add_scalar('Validation/Loss', val_loss/len(val_loader), epochs)
        writer.add_scalar('Validation/Accuracy', valid_accuracy, epochs)
        writer.add_scalar('Validation/Precision', valid_precision, epochs)
        writer.add_scalar('Validation/Recall', valid_recall, epochs)
        writer.add_scalar('Validation/F1 Score', valid_f1, epochs)    

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        if val_loss/len(val_loader) < best_val_loss:
            best_val_loss = val_loss/len(val_loader)
    writer.close()
    torch.save(model.state_dict(), config['paths']['model_save_path'])

if __name__ == "__main__":
    training()