
import sys
import torch
import torchvision
import argparse
from tqdm import tqdm
from models.select_model import *
from utils.dataset import *
from utils.log_args import Instructor


########## Hyper Parameters #######


def train(self, train_dataloader, criterion, optimizer):
    train_loss, n_correct, n_train = 0, 0, 0
    n_batch = len(train_dataloader)
    self.model.train()
    for i_batch, (inputs, targets) in enumerate(tqdm(train_dataloader, 0)):
        inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)  # Move the input data to the GPU
        def closure():
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
            return outputs, loss
        outputs, loss = optimizer.step(closure)
        # train_loss += loss.item() * targets.size(0)
        train_loss += loss.item()
        n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
        # n_correct += (torch.max(outputs.data, 1) == targets).sum().item()
        n_train += targets.size(0)
        ratio = int((i_batch+1)*50/n_batch)
        sys.stdout.write(f"\r[{'>'*ratio}{' '*(50-ratio)}] {i_batch+1}/{n_batch} {(i_batch+1)*100/n_batch:.2f}%")
        sys.stdout.flush()              
        if i_batch % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (self.args.num_epoch + 1, i_batch + 1, train_loss / 100))
            train_loss = 0.0  
    print()
    return train_loss / n_train, n_correct / n_train

def val(self, test_dataloader, criterion):
    test_loss, n_correct, n_test = 0, 0, 0
    n_batch = len(test_dataloader)
    self.model.eval()
    with torch.no_grad():
        for i_batch, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            # print(f"{i_batch}th, targets.size(0): {targets.size(0)}")                 
            # n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            test_loss += loss.item()
            n_test += targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            n_correct += (predicted == targets).sum().item()
            ratio = int((i_batch+1)*50/n_batch)
            sys.stdout.write(f"\r[{'>'*ratio}{' '*(50-ratio)}] {i_batch+1}/{n_batch} {(i_batch+1)*100/n_batch:.2f}%")
            sys.stdout.flush()                
    print( '\nAccuracy of the network on the 10000 test images: %f %%' % (100 * n_correct / n_test))
    print('Finished Training')

    return test_loss / n_test, n_correct / n_test


# def validate(model, criterion, dataloader, device):
#     model.eval()

#     running_valid_loss = 0.0
#     total_valid_predictions = 0.0
#     correct_valid_predictions = 0.0
#     all_valid_labels = []
#     all_valid_predictions = []

#     with torch.no_grad():
#         for data, target in dataloader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             loss = criterion(output, target)
#             running_valid_loss += loss.item() * data.size(0)

#             _, predicted = torch.max(output.data, 1)
#             total_valid_predictions += target.size(0)
#             correct_valid_predictions += (predicted == target).sum().item()

#             all_valid_labels.extend(target.detach().cpu().numpy().tolist())
#             all_valid_predictions.extend(predicted.detach().cpu().numpy().tolist())

#     valid_loss = running_valid_loss / len(dataloader.dataset)
#     valid_acc = (correct_valid_predictions / total_valid_predictions) * 100.0
#     valid_f1 = f1_score(all_valid_labels, all_valid_predictions, average='macro')

#     return valid_loss, valid_acc, valid_f1, data, output, target


