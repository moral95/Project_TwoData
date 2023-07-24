
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from utils.dataset import McDataset
from utils.config import load_config
from sklearn.model_selection import train_test_split
from torch import binary_cross_entropy_with_logits # 해당 데이터셋은 binary가 아니다. 다른 걸로 보도록
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True   # 해당 코드는 무엇?
    torch.backends.cudnn.benchmarks = True  # 해당 코드는 무엇?

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
# print(num_features)

class Bert(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.num_feartures = num_features
        self.classifier = nn.Softmax()
        # self.classifier = nn.Sequential(
        #                         nn.Linear(num_features, num_features//2),
        #                         nn.ReLU(),
        #                         nn.Linear(num_features//2, 5),
        #                         nn.Softmax(),)
        
    def forward(self, x):
        x= self.model(x)
        output = self.classifier(x)

        return output



model.classifier = nn.Sequential(
    nn.Linear(num_features, num_features//2),
    nn.ReLU(),
    nn.Linear(num_features//2, 5),
    nn.Softmax(),
    # nn.AdaptiveAvgPool1d()
    )
    
# model = Bert(model)
# print(model) 
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr= lr)
criterion = torch.nn.CrossEntropyLoss()

######### DATA SET ########
train_dataset = McDataset(annotation_file, tokenizer, mode='train', seed= seed, max_len=max_len)
train_loader = DataLoader(train_dataset, batch_size= config['training']['batch_size'], shuffle = True)
print(f'train_dataset : {len(train_dataset)}')

val_dataset = McDataset(annotation_file, tokenizer, mode='val', seed= seed, max_len=max_len)
val_loader = DataLoader(val_dataset, config['training']['batch_size'], shuffle = False)
print(f'val_dataset : {len(val_dataset)}')

######### train, val def ######

def train(model, criterion, optimizer, dataloader, device, epochs):
    model.train()
    
    epoch_loss = 0
    running_loss, total_predictions, correct_predictions = 0, 0, 0
    all_predictions = []
    all_labels = []
    for ids, masks, targets  in tqdm(dataloader):
        optimizer.zero_grad()

        input_ids = ids.to(device)
        # print(f'input_ids.dtype : {input_ids.dtype}')
        # print(f'input_ids.shape : {input_ids.shape}')
        # print(f'input_ids : {input_ids}')
        attention_mask = masks.to(device)
        # print(f'masks.dtype : {masks.dtype}')
        # print(f'masks.shape : {masks.shape}')
        # print(f'masks : {masks}')
        labels = targets.to(device)
        # print(f'label.dtype : {labels.dtype}')
        # print(f'labels.shape : {labels.shape}')
        # print(f'labels : {labels}')


        outputs = model(input_ids, attention_mask=attention_mask)
        # outputs = outputs.logits.max(1)[0]
        # print(f'output.dtype : {outputs.dtype}')
        outputs = outputs.logits.max(1)[1]
        outputs = torch.tensor(outputs, dtype = torch.float)
        # print(f'output.dtype : {outputs.dtype}')
        
        # print(f'outputs.logits: {outputs.logits.shape}')
        # print(f'outputs.logits: {outputs.shape}')
        # print(f'outputs: {outputs}')
        # print(f'labels : {labels.shape}')

        # loss = criterion(outputs.logits, labels)
        loss = criterion(outputs, labels).to(device)
        loss.requires_grad_(True)
        # running_loss += loss.item() * input_ids.size(0)
        # print(f'loss.shape : {loss.shape}')
        # print(f'loss : {loss}')
        # print(f"inputs_ids.size(0) : {input_ids.size(0)}")
        running_loss += loss.item() * input_ids.size(0)
        # 배치사이즈를 곱해준다.
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        # epoch_loss += loss.item() * ids.size(0)
        # print(f'epoch_loss : {epoch_loss}')
        # print(f'data.size(0) : {input_ids.size(0)}')

        _, predict = torch.max(outputs.data, 0)
        total_predictions +=  labels.size(0)
        # print(f'total_predictions : {total_predictions}')
        # print(f'predict : {predict}')
        correct_predictions += (predict == labels).sum().item()

        # preds = torch.argmax(outputs, dim=1)
        # _, preds = torch.max(outputs.data, dim=1)
        # total_predictions += labels.size(0)
        # train_acc += preds.eq(labels).sum().item()
        all_predictions.append(outputs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        
        # all_labels.extend(labels.detach().cpu().numpy().tolist())
        # all_predictions.extend(preds.detach().cpu().numpy().tolist())


    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    train_precision = precision_score(all_labels, all_predictions, average='weighted')
    train_recall = recall_score(all_labels, all_predictions, average='weighted')
    train_f1 = f1_score(all_labels, all_predictions, average='weighted')
    train_accuracy = accuracy_score(all_labels, all_predictions)
    train_confusion_matrix = confusion_matrix(all_labels, all_predictions)

    print(f'len_dataloader : {len(dataloader)}')
    print(f'len_dataloader.dataset : {len(dataloader.dataset)}')

    print(f'train_acc : {correct_predictions / total_predictions}')
    print(f'train_loss : {epoch_loss / len(dataloader)}')
    print(f'train_loss_dataset : {epoch_loss / len(dataloader.dataset)}')


    return train_precision, train_recall, train_f1, train_accuracy, train_confusion_matrix, epoch_loss

def val(model, criterion, dataloader, device, epochs):
    model.eval()

    running_valid_loss, total_valid_predictiosn, correct_valid_predictions = 0, 0, 0
    val_loss = 0
    all_valid_labels, all_valid_predictions = [], []

    with torch.no_grad():
        for ids, mask, label in tqdm(dataloader):
            input = ids.to(device)
            mask = mask.to(device)
            label = label.to(device)

            output = model(input, attention_mask = mask)
            outputs = output.logits.max(1)[1]
            outputs = torch.tensor(outputs, dtype = torch.float)            
            loss = criterion(outputs, label)
            loss.requires_grad_(True)

            val_loss += loss.item()
            all_valid_predictions.append(outputs.detach().cpu().numpy())
            all_valid_labels.append(label.detach().cpu().numpy())            

    all_predictions = np.concatenate(all_valid_predictions)
    all_labels = np.concatenate(all_valid_labels)
    
    valid_precision = precision_score(all_labels, all_predictions, average='weighted')
    valid_recall = recall_score(all_labels, all_predictions, average='weighted')
    valid_f1 = f1_score(all_labels, all_predictions, average='weighted')
    valid_accuracy = accuracy_score(all_labels, all_predictions)
    valid_confusion_matrix = confusion_matrix(all_labels, all_predictions)

    return valid_precision, valid_recall, valid_f1, valid_accuracy, valid_confusion_matrix, val_loss



def training(config):
    
    print(f'We use the {device}')

    for epochs in range(num_epochs):
        train_precision, train_recall, train_f1, train_accuracy, train_confusion_matrix, epoch_loss = train(model, criterion, optimizer, train_loader, device, epochs)
        valid_precision, valid_recall, valid_f1, valid_accuracy, valid_confusion_matrix, val_loss = val(model, criterion, val_loader, device, epochs)

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

    if val_loss/len(val_loader) < best_val_loss:
        best_val_loss = val_loss/len(val_loader)
        # torch.save(model.state_dict(), config['paths']['model_save_path'])
        torch.save(model.state_dict(), config['paths']['model_save_path'])
    writer.close()

if __name__ == "__main__":
    config = load_config("configs/configs.yaml")
    training(config)