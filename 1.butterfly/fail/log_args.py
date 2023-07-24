# ./utils/log_args.py

import sys
import sys
import torch
import logging
from models.select_model import *


class Instructor:

    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.addHandler(logging.FileHandler(args.log_name))
        # self.model = model_selection(self.args.model, self.args.num_classes)
        # self.model.to(args.device)
        # if args.device.type == 'cuda':
        #     self.logger.info(f'>> cuda memory allocated : {torch.cuda.memory_allocated(args.device.index)}')
        # self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.size()))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
            self.logger.info(f"> n_trainable_params : {n_trainable_params}, n_nontrainable_params: {n_nontrainable_params}")
            self.logger.info('>training arguments:')
            for arg in vars(self.args):
                self.logger.info(f">>> {arg} : {getattr(self.args, arg)}")