# ./utils/dataset.py
import os
import pandas as pd
import re
import torch
from torch.utils.data import Dataset
from utils.config import load_config
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")
  
class McDataset(Dataset):
    def __init__(self, annotations_file, tokenizer, mode, seed, max_len):
        self.config = load_config('configs/configs.yaml')
        self.data_df = pd.read_csv(annotations_file, encoding='latin-1')[['review','rating']].dropna(axis=0)
        self.tokenizer = tokenizer
        self.data_train, self.data_temp = train_test_split(self.data_df, test_size=0.4, random_state= seed)
        self.data_valid, self.data_test = train_test_split(self.data_temp, test_size = 0.5, random_state = seed)
        self.max_len = max_len

        if mode == 'train':
            self.mode = self.data_train
        elif mode == 'val':
            self.mode = self.data_valid
        elif mode == 'test':
            self.mode = self.data_test
        else:
            print(f'{mode} is NOT suitable Mode')

    def __len__(self):
        return len(self.mode)
    
    def __getitem__(self, idx):

        #### 전처리 구간 ####
        self.mode = self.mode.reset_index(drop=True)
        self.data_review = self.mode['review'].str.encode("ascii", "ignore").str.decode("utf-8")
        data_review = self.data_review.iloc[idx]
        data_review = " ".join(data_review.split())        


            ######## 1 :  hugging transform ######
        # encoded_input = tokenizer(data_review)
        # encoded_review = encoded_input['input_ids']

            ######## 2 : tensorflow.karas #########
        # tokenizer = self.tokenizer(num_words = self.vocab_length)
        # tokenizer.fit_on_texts(data_review)
        # token_data = tokenizer(data_review)
        # print(f"Tokenizer 후 : {token_data}")

            ######## 3 : hugging transform (plus) #####
        inputs = self.tokenizer.encode_plus(
            data_review,
            None,
            add_special_tokens = True,
            max_length = self.max_len,
            pad_to_max_length = True,
            return_token_type_ids = True,
            truncation = True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        label = self.mode['rating'][idx][0]

        ids = torch.tensor(ids, dtype = torch.long)
        mask = torch.tensor(mask, dtype = torch.long)
        targets = torch.tensor(int(label), dtype=torch.float)

        return ids, mask, targets
