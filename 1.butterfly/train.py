
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import torch.nn as nn

def train(model, device, dataloader, criterion, optimizer, config):
    model.to(device)

    running_loss, total_predictions, correct_predictions = 0, 0, 0
    all_labels, all_predictions = [], []

    model.train()
    print(dataloader)
    for data, label in tqdm(dataloader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, label)
        # Gradient clipping
        if config['training']['gradient_clipping']:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5, norm_type=1.0)

        running_loss += loss.item() * data.size(0)

        loss.backward()
        optimizer.step()

        _, predict = torch.max(output.data, 1)

        total_predictions += label.size(0)
        # 예측치 -> 배치의 수만큼 예측한 것들 (개수로 입력)
        correct_predictions += (predict == label).sum().item()
        # 제대로 맞춘 예측치 -> 전체 배치 안에서 맞춘 것들 (개수로 입력)

        all_labels.extend(label.detach().cpu().numpy().tolist())
        all_predictions.extend(predict.detach().cpu().numpy().tolist())

    epoch_loss = running_loss/ len(dataloader.dataset)
    epoch_acc = (correct_predictions/ total_predictions) * 100

    train_precision = precision_score(all_labels, all_predictions, average='macro')
    train_recall = recall_score(all_labels, all_predictions, average='macro')
    train_f1 = f1_score(all_labels, all_predictions, average='macro')
    train_accuracy = accuracy_score(all_labels, all_predictions)
    train_confusion_matrix = confusion_matrix(all_labels, all_predictions)


    return epoch_loss, epoch_acc, train_precision, train_recall, train_f1, train_accuracy, train_confusion_matrix

def validate(model, device, dataloader, criterion):
    model.eval()

    running_valid_loss = 0.0
    total_valid_predictions = 0.0
    correct_valid_predictions = 0.0
    all_valid_labels = []
    all_valid_predictions = []

    with torch.no_grad():
        for data, target in tqdm(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_valid_loss += loss.item() * data.size(0)

            _, predicted = torch.max(output.data, 1)
            total_valid_predictions += target.size(0)
            correct_valid_predictions += (predicted == target).sum().item()

            all_valid_labels.extend(target.detach().cpu().numpy().tolist())
            all_valid_predictions.extend(predicted.detach().cpu().numpy().tolist())

    valid_loss = running_valid_loss / len(dataloader.dataset)
    valid_acc = (correct_valid_predictions / total_valid_predictions) * 100.0

    valid_precision = precision_score(all_valid_labels, all_valid_predictions, average='macro')
    valid_recall = recall_score(all_valid_labels, all_valid_predictions, average='macro')
    valid_f1 = f1_score(all_valid_labels, all_valid_predictions, average='macro')
    valid_accuracy = accuracy_score(all_valid_labels, all_valid_predictions)
    valid_confusion_matrix = confusion_matrix(all_valid_labels, all_valid_predictions)

    return valid_loss, valid_acc, valid_precision, valid_recall, valid_f1, valid_accuracy, valid_confusion_matrix



