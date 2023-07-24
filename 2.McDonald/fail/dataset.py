# ./utils/dataset.py
import os
import pandas as pd
import re
import torch
from torch.utils.data import Dataset
from utils.config import load_config
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import warnings
warnings.filterwarnings("ignore")

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(), !?\'\']"," ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower()       

def Tokenizer(tokenizer, data_review, max_len):
            inputs = tokenizer.encode_plus(
            data_review,
            None,
            add_special_tokens = True,
            max_length = max_len,
            pad_to_max_length = True,
            return_token_type_ids = True,
            truncation = True
        )
            return inputs

class McDataset(Dataset):
    def __init__(self, annotations_file, tokenizer, mode, seed, max_len):
        self.config = load_config('configs/configs.yaml')
        self.data_df = pd.read_csv(annotations_file, encoding='latin-1')[['review','rating']].dropna(axis=0)
        self.tokenizer = tokenizer
        self.data_train, self.data_test = train_test_split(self.data_df, test_size=0.3, random_state= seed)
        # self.vocab_lenth = vocab_length
        self.max_len = max_len

        if mode == 'train':
            self.mode = self.data_train
        elif mode == 'val' or 'test':
            self.mode = self.data_test
        else:
            print(f'{mode} is NOT suitable Mode')

        # print(self.mode)
        # print(self.data_review)
        # print(len(self.data_review))

    def __len__(self):
        return len(self.mode)
    
    def __getitem__(self, idx):
        # data_review = self.data_df.iloc[8][idx]

        #### 전처리 구간 ####
        # print(f'텍스트 정제 전 : {self.data_review}')
        # self.data_review = self.data_review.dropna()
        # data_review_c = clean_str(self.data_review[idx])
        self.mode = self.mode.reset_index(drop=True)
        self.data_review = self.mode['review'].str.encode("ascii", "ignore").str.decode("utf-8")
        data_review = self.data_review.iloc[idx]
        # print(f'data_review : {data_review}')
        # data_review_c = [clean_str(x) for x in self.data_review]
        # data_review_c = [clean_str(x) for x in self.data_review].dropna(axis=0)
        # for x in self.data_review:
        #     data_review_c = clean_str(x)
        data_review = " ".join(data_review.split())        
            
        # print(f'텍스트 정제 후 :{data_review_c[idx]}')
        # data_review_cc = " ".join(data_review_cc.split())
        # print(f'텍스트 분리 후 : {data_review[:4]}')

        # max length
        # self.data_review.str.len().hist()
        # plt.xlabel('Review Length')
        # plt.ylabel('Frequency')
        # plt.show()
        # max_len = 1000

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
        # print(inputs)

        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        # print(f'mode : {self.mode}')
        # rating = self.mode['rating']
        # print(f'rating : {rating}')
        label = self.mode['rating'][idx][0]
        # print(f'label : {label}')
        # targets = targets[idx][0]
        # print(f'target : {targets}')
        # ids = torch.tensor(ids, dtype = torch.int64)
        # mask = torch.tensor(mask, dtype = torch.int64)
        ids = torch.tensor(ids, dtype = torch.long)
        mask = torch.tensor(mask, dtype = torch.float)
        # targets = torch.tensor(int(label), dtype=torch.int64)
        targets = torch.tensor(int(label), dtype=torch.float)
        # targets = torch.tensor(int(label))
        return ids, mask, targets
