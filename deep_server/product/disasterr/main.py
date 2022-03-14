
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
import seaborn as sns
from collections import defaultdict
import time
import pandas as pd
import matplotlib.pyplot as plt
import os

from .dataset import Dataset
from .train import train
from .show_loss import plot_loss, plot_acc
from .inference import predict

#pip install git+https://github.com/huggingface/transformers#
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup



#Parmeters and Settings#
def run(url):
    num_Epochs = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path='/home/ec2-user/yuddomack/product/disasterr/'
    save_model_path='/home/ec2-user/yuddomack/product/disasterr/checkpoint/'
    save_inference_path='/home/ec2-user/yuddomack/product/disasterr/checkpoint/'
    save_model_name='BERT'
    percent_of_train=0.9
    shuffle_train_data=True
    Num_labels=2
    Learning_Rate=2e-5
    print("device", device)



    print("PATH", path)
    #train_csv = pd.read_csv(os.path.join(path+'train.csv'), keep_default_na = False)
    train_csv = pd.read_csv(url, keep_default_na = False)
    print(train_csv.head())
    if shuffle_train_data :
        train_csv = train_csv.sample(frac=1).reset_index(drop=True)
    num_of_train=round(percent_of_train*(len(train_csv)))
    train_data = train_csv.iloc[:num_of_train]
    valid_data = train_csv.iloc[num_of_train:]
    print("Import Data")

    '''
    Dataloader
    '''
    train_dataset = Dataset(train_data, name = 'train')
    train_dataloader = data.DataLoader(train_dataset, shuffle= True, batch_size = 32, num_workers=16,pin_memory=True)

    validation_dataset = Dataset(valid_data, name = 'valid')
    validation_dataloader = data.DataLoader(validation_dataset, shuffle= True, batch_size = 32, num_workers=16,pin_memory=True)

    test_csv = pd.read_csv(os.path.join(path+'test.csv'), keep_default_na = False)
    test_dataset = Dataset(test_csv , name = 'test')
    test_dataloader = data.DataLoader(test_dataset, shuffle= False, batch_size = 1, num_workers=16,pin_memory=True)

    '''
    Model
    '''
    Bert_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", #12-layer BERT model, with an uncased vocab.
        num_labels = Num_labels, #Number of Classes
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    Bert_model.to(device)
    print("Import Model")


    '''
    Train
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(Bert_model.parameters(), lr = Learning_Rate,eps = 1e-8)
    total_steps = len(train_dataloader) * num_Epochs
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,  num_training_steps = total_steps)
    modelpath= os.path.join(save_model_path,save_model_name)
    train_loss, valid_loss, valid_acc = train(Bert_model, train_dataloader, validation_dataloader, criterion, optimizer, lr_scheduler, modelpath, device, epochs = num_Epochs)

    '''
    Plot Loss
    '''
    print("FINISH TRAIN")
    
    """
    sns.set_style("whitegrid")
    plot_loss(num_Epochs, train_loss, valid_loss, title='Loss plot')
    plot_acc(num_Epochs, valid_acc)

    '''
    Inference
    '''
    predict(Bert_model, test_dataloader, device,save_inference_path)
    """