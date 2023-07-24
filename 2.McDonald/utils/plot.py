# ./utils/plot.py

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

data_df = pd.read_csv("/NasData/home/lsh/10.project/data/mcdonalds/McDonald_s_Reviews.csv", encoding='unicode_escape')

def plot_dataset_distribution():
    
    data_df['rating'].value_counts().plot(kind='bar')

    plt.title('rating distribution')
    plt.xlabel('rating')
    plt.ylabel('Count')

    for i in range(len(data_df['rating'].value_counts())):
        plt.text(i, data_df['rating'].value_counts()[i], data_df['rating'].value_counts()[i], ha='center', va='bottom')
    
    plt.savefig('/NasData/home/lsh/10.project/2.McDonald/savefig/savefig_data_distribution.png')
    plt.close()

def plot_data_length():

    data_df['review'].str.len().hist()

    plt.xlabel('Review Length')
    plt.ylabel('Frequency')
    plt.savefig('/NasData/home/lsh/10.project/2.McDonald/savefig/savefig_review_length.png')
    plt.close()

def plot_split_data_length():

    data_train, data_temp = train_test_split(data_df, test_size=0.4, random_state= 5657)
    data_valid, data_test = train_test_split(data_temp, test_size = 0.5, random_state = 5657)

    fig, ax = plt.subplots()

    dataset = ['train', ' valid', 'test']
    counts = [data_train['rating'].count(), data_valid['rating'].count(), data_test['rating'].count()]
    bar_labels = ['red', 'blue', 'orange']
    bar_colors = ['tab:red', 'tab:blue', 'tab:orange']

    ax.bar(dataset, counts, label=bar_labels, color=bar_colors)
    ax.set_ylabel('Dataset length')
    ax.set_title('Dataset length with mode')
    ax.legend(title='mode')

    plt.savefig('/NasData/home/lsh/10.project/2.McDonald/savefig/savefig_split_Data_length.png', )
    plt.close()
