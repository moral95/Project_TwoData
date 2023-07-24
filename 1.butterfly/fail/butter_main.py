
import sysS
import torch
import torchvision
import argparse
import datetime

from utils.log_args import Instructor
from utils.dataset import *
from models.select_model import *
# from butter_train import train_one_epoch, validate

from butter_train import train, val

########## Hyper Parameters #######

def run(self):
    # train_dataloader, test_dataloader =  getdata(resize=self.args.resize,
    #                                             batch_size=self.args.batch_size,
    #                                             dataset=self.args.dataset,
    #                                             data_target_dir=os.path.join(self.args.data_dir, self.args.dataset),
    #                                             data_aug=(self.args.no_data_aug==False),
    #                                             cutout=self.args.cutout,
    #                                             autoaug=self.args.autoaug)

    # config = load_config('configs/configs.yaml')  # specify the path to your config file
    train_dataloader, valid_dataloader, test_dataloader = data_loader()
    model = model_selection(self.args.model, self.args.num_classes)
    model.to(args.device)
    criterion = criterion_selection(self.args.criterion)
    optimizer = optim_selection(self.args.optim, model, self.args.lr)
    scheduler = scheduler_selection(self.args.scheduler, optimizer, 10)

    best_loss, best_acc = 0, 0
    for epoch in range(self.args.num_epoch):
        train_loss, train_acc = train(self, train_dataloader=train_dataloader, criterion = criterion, optimizer=optimizer)
        test_loss, test_acc = val(test_dataloader, criterion)
        # scheduler.step(train_loss/100)
        scheduler.step()
        if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
            best_acc, best_loss = test_acc, test_loss
        # Save the checkpoint of the last model
        self.logger.info(f"{epoch+1}/{self.args.num_epoch} - {100*(epoch+1)/self.args.num_epoch:.2f}%")
        self.logger.info(f"[train] loss: {train_loss:.4f}, acc: {train_acc*100:.2f}, err: {100-train_acc*100:.2f}")
        self.logger.info(f"[test] loss: {test_loss:.4f}, acc: {test_acc*100:.2f}, err: {100-test_acc*100:.2f}")
    self.logger.info(f"best loss: {best_loss:.4f}, best acc: {best_acc*100:.2f}, best err: {100-best_acc*100:.2f}")
    self.logger.info(f"log saved: {self.args.log_name}")
    PATH = './vit_cifar10_%f_%d.pth' % (self.args.lr, self.args.seed)
    torch.save(self.args.model.state_dict(), config['paths']['model_save_path'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default = 'resnet50', choices='model_names', help= 'Model architecture.')
    parser.add_argument('--optim', default = 'Adam', choices='optimizer_names', help= 'Optimizer.')
    parser.add_argument('--scheduler', default = 'StepLR', choices='scheduler_names', help= 'Scheduler.')
    parser.add_argument('--criterion', default = 'CEL', choices='criterion_names', help= 'Criterion.')
    parser.add_argument('--num_classes', type = int, default = '75', help= 'Dataset name.')

    parser.add_argument('--num_epoch', type = int, default = '200', help= 'Dataset name.')
    parser.add_argument('--batch_size', type=int, default=128, help= 'batch_size')
    parser.add_argument('--gpus', type=str, default='0', help= 'epochs')
    parser.add_argument('--seed', type=int, default= '7890', help= 'seed')
    
    parser.add_argument('--resize', type=int, default=224, help= 'resize')

    parser.add_argument('--lr', type=float, default=1e-5, help= 'lr')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help= 'weight_decay')

    parser.add_argument('--clip_norm', type=int, default=50, help='Maximum norm of parameter gradient.')

    # parser.add_argument('--autoaug', default=False, action='store_true', help='Enable AutoAugment.')

    parser.add_argument('--default_directory', type=str, default= '/NasData/home/lsh/10.project/butterfly/save', help= 'default_directory')

    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'], help='Device.')

    args = parser.parse_args()
    args.log_name = f"{args.model}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:]}.log"
    args.device = torch.device(args.device) if args.device else torch.device('cuda:2' if torch.cuda.is_available else 'cpu')
    ins = Instructor(args)
    run(ins)