import torch
import torch.nn as nn

from datasets import load_dataset
import numpy as np
import os
#####################
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from bart_dataset import IMDBDataset
#####################
from transformers import BartTokenizer
#####################
from bart_model import TestModel 
from tqdm import tqdm
#####################

if torch.cuda.is_available():
    device = "cuda:2"
else:
    device = "cpu"   

device = torch.device(device)

from datasets import load_dataset

print('load imdb')
dataset = load_dataset("imdb")

test_dat = dataset['test']


print('tokenizer')
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

test_encoding = tokenizer(
    test_dat['text'],
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length = 512
)

test_len = int(len(test_dat['text']))
print('dataloader')    
test_set = IMDBDataset(test_encoding, test_dat['label'])
test_loader = DataLoader(test_set, batch_size=16, shuffle=True)

print('load the best trained model')
model = TestModel()
PATH = os.getcwd() + '/check_point/best_model.pt'
PATH = '/home/sangmin/classification_test/check_point/best_model.pt'

model.load_state_dict(torch.load(PATH))
def evaluate(model, testloader, device):
    model.to(device)
    model.eval()
    
    total_acc_test = 0

    with torch.no_grad():
        
        for i, test_input in enumerate(tqdm(testloader)):

            test_label = test_input['labels'].to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print('='*64)        
    print(f'Train Accuracy: {total_acc_test / test_len: .3f}')
    print('='*64)
    
print("evaluate")
evaluate(model, test_loader, device)