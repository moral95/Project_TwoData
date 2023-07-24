# ./utils/plot.py

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from utils.config import load_config
from sklearn.model_selection import train_test_split

#############################################################################

def plot_dataset_distribution(data):
    annotation = pd.read_csv(data)
    
    annotation['label'].value_counts().plot(kind='bar')

    plt.title('label distribution')
    plt.xlabel('label')
    plt.ylabel('Count')

    for i in range(len(annotation['label'].value_counts())):
        plt.text(i, annotation['label'].value_counts()[i], annotation['label'].value_counts()[i], ha='center', va='bottom')
    plt.savefig('/NasData/home/lsh/10.project/1.butterfly/savefig/butter_data_distribution.png')
    plt.close()

#############################################################################

def plot_data_len():
    temp_df = pd.read_csv("/NasData/home/lsh/10.project/data/butterfly/Training_set.csv")
    test_df = pd.read_csv("/NasData/home/lsh/10.project/data/butterfly/Testing_set.csv")

    data_train, data_valid = train_test_split(temp_df, test_size=0.2, random_state= 5657)
    data_test = test_df

    fig, ax = plt.subplots()
    dataset = ['train', ' valid', 'test']
    counts = [data_train['filename'].count(), data_valid['filename'].count(), data_test['filename'].count()]
    bar_labels = ['red', 'blue', 'orange']
    bar_colors = ['tab:red', 'tab:blue', 'tab:orange']
    ax.bar(dataset, counts, label=bar_labels, color=bar_colors)
    ax.set_ylabel('Dataset length')
    ax.set_title('Dataset length with mode')
    ax.legend(title='mode')
    plt.savefig('/NasData/home/lsh/10.project/1.butterfly/savefig/butter_data_len.png')
    plt.close()

#############################################################################

def plot_image_sample(dataset_paths, image_paths):
    image_df = pd.read_csv(image_paths)
    train_img_filename = image_df['filename']
    image = os.path.join(dataset_paths,'train', train_img_filename[3])
    img_sample = cv2.imread(image)
    plt.imshow(cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB))
    plt.savefig('/NasData/home/lsh/10.project/1.butterfly/savefig/butter_image_sample.png')
    plt.close()

#############################################################################

def plot_image_samplesx4(dataset_path, train_path):
    fig = plt.figure()
    rows = 2
    cols = 2
    a = 1

    xlabels = ["xlabels", "(a)", "(b)", "(c)", "(d)"]

    sample_show = 4
    for i in range(sample_show):
        annotation = pd.read_csv(train_path)
        file_name = annotation['filename']
        img_path = os.path.join(dataset_path, 'train/', file_name[i])    # data_df.iloc[idx, 0]
        img_sample = cv2.imread(img_path)
        ax = fig.add_subplot(rows, cols, a)
        ax.imshow(cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB))
        ax.set_xlabel(xlabels[a])
        a += 1
        ax.set_xticks([]), ax.set_yticks([])
    plt.savefig('/NasData/home/lsh/10.project/1.butterfly/savefig/butter_image_samplesx4.png')    
    plt.close()

#############################################################################
