
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
from .inference import predict_input

#pip install git+https://github.com/huggingface/transformers#
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

def run_test():

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        path='/home/ec2-user/yuddomack/product/disasterr/'
        save_model_path='/home/ec2-user/yuddomack/product/disasterr/checkpoint/'
        save_inference_path='/home/ec2-user/yuddomack/product/disasterr/checkpoint/'
        save_model_name='BERT'
        Num_labels=2
        Learning_Rate=2e-5
        print("device", device)
        modelpath= os.path.join(save_model_path,save_model_name)

        #test_csv = pd.read_csv(os.path.join(path+'test.csv'), keep_default_na = False)
        #test_dataset = Dataset(test_csv , name = 'test')
        #test_dataloader = data.DataLoader(test_dataset, shuffle= False, batch_size = 1, num_workers=16,pin_memory=True)
        input = "Awesome!"
        Bert_model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", #12-layer BERT model, with an uncased vocab.
            num_labels = Num_labels, #Number of Classes
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        Bert_model.load_state_dict(torch.load(modelpath)['model_state_dict'])
        Bert_model.to(device)
        print("Import Model")

        target = predict_input(Bert_model, input, device, save_inference_path)
        
        return target[0]