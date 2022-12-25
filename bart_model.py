import torch
import torch.nn as nn
from transformers import BartModel

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        
        #bart encoder
        self.base_model = BartModel.from_pretrained("facebook/bart-base")
        self.enc = self.base_model.encoder

        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 2)
        # self.linear = nn.Linear(768,2)
        self.dropout = nn.Dropout(0.3)
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, input_ids, attention_mask):
        
        #bart
        outputs = self.enc(input_ids, attention_mask) # (batch, n_seq, dimensions)
        
        # You write you new head here
        last_hidden_states = outputs.last_hidden_state
        outputs = torch.sum(last_hidden_states, dim=1)
        
        out = self.dropout(self.leaky_relu(self.linear1(outputs)))
        out = self.dropout(self.leaky_relu(self.linear2(out)))
        linear_out = self.linear3(out)
        
        out = self.leaky_relu(linear_out)
        
        return out