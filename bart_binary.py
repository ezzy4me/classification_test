import torch
import torch.nn as nn

from datasets import load_dataset
import os
import numpy as np
#####################
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from bart_dataset import IMDBDataset
#####################
from transformers import AdamW
from transformers import get_scheduler
from transformers import BartTokenizer
#####################
from bart_model import TestModel 
from tqdm import tqdm
#####################
from torch.optim import AdamW

PATH = os.getcwd() + '/check_point/best_model.pt'
PATH = '/home/sangmin/classification_test/check_point/best_model.pt'

if torch.cuda.is_available():
    device = "cuda:2"
else:
    device = "cpu"   

device = torch.device(device)

from datasets import load_dataset

print('load imdb')
dataset = load_dataset("imdb")

train_dat = dataset['train']
test_dat = dataset['test']


print('tokenizer')
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

train_encoding = tokenizer(
    train_dat['text'][:int(0.8*len(train_dat['text']))],
    return_tensors='pt',
    padding=True,
    truncation=True
)

val_encoding = tokenizer(
    train_dat['text'][int(0.8*len(train_dat['text'])):],
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length = 512
)

test_encoding = tokenizer(
    test_dat['text'],
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length = 512
)

train_len = int(0.8*len(train_dat['text']))
val_len = int(0.2*len(train_dat['text']))
test_len = int(len(test_dat['text']))


print('dataloader')    
train_set = IMDBDataset(train_encoding, train_dat['label'][:train_len])
val_set = IMDBDataset(val_encoding, train_dat['label'][train_len:])
test_set = IMDBDataset(test_encoding, test_dat['label'])

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=True)
test_loader = DataLoader(test_set, batch_size=8, shuffle=True)

from tqdm import tqdm

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
model = TestModel()

print("train loop")
def train(epoch, model, trainloader, valloader, optimizer, device):
    model.to(device)
    
    # for torch.save()
    best_acc_model = None
    best_acc = 0
    
    for e in range(1, epoch+1):

        total_acc_train = 0
        total_loss_train = 0
        
        model.train()
        for i, train_input in enumerate(tqdm(trainloader)):
            
            train_label = train_input['labels'].to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            
            output = model(input_id, mask)
            
            batch_loss = criterion(output.contiguous(), train_label.long())
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

        model.eval()
        with torch.no_grad():
            
            for i, val_input in enumerate(tqdm(valloader)):

                val_label = val_input['labels'].to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
                
            if total_acc_val > best_acc:
                best_acc = total_acc_val
                best_acc_model = torch.save(model.state_dict(), PATH)

        print('='*64)
        # print(
        #     f'Epochs: { e } | Train Loss: {total_loss_train / train_len: .3f} \
        #     | Train Accuracy: {total_acc_train / train_len: .3f}')
        print(
            f'Epochs: { e } | Train Loss: {total_loss_train / train_len: .3f} \
            | Train Accuracy: {total_acc_train / train_len: .3f} \
            | Val Loss: {total_loss_val / val_len: .3f} \
            | Val Accuracy: {total_acc_val / val_len: .3f}')
        print('='*64) 
       
# ================= implement ================= 
print("train") 
optimizer = AdamW(model.parameters(), lr=5e-5)
train(10, model, train_loader, val_loader, optimizer, device)
