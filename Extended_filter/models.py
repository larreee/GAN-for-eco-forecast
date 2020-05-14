import torch
import torch.nn as nn
import random

class LSTMGenerator(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, horizon, \
                 init_batch_size = 1, device= torch.device('cuda:0')):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.device = device
        self.horizon = horizon

        self.filter = nn.Sequential(nn.Linear(768,768), nn.Linear(768,768))
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers = num_layers, batch_first = True).to(self.device)
        self.linear = nn.Linear(hidden_layer_size, output_size).to(self.device)
        self.hidden_cell = (torch.zeros(num_layers, init_batch_size, self.hidden_layer_size).to(self.device),
                            torch.zeros(num_layers, init_batch_size, self.hidden_layer_size).to(self.device))
    
    def forward(self, cat, input_):
        batch_z = input_[1].size(0)
        seq_length = input_[1].size(1)
        # noise_, seq, enc = input_
        encoded_list = []
        for b in input_[2]:
            batch_list = []
            for week in b.index:
                week_encs = b.loc[week]
                week_filtered = self.filter(week_encs.to(self.device))
                week_mean = week_filtered.mean(0).unsqueeze(0)
                batch_list.append(week_mean)
            batch_enc = torch.cat(batch_list)
            encoded_list.append(batch_enc)
        encoded = torch.stack(encoded_list)

        lstm_out, self.hidden_cell = self.lstm(cat(input_[0], input_[1], encoded), self.hidden_cell)
        predictions = self.linear(lstm_out.view(-1, seq_length, self.hidden_layer_size)).to(self.device)
        del lstm_out
        return predictions[:,:self.horizon,:], encoded

    def clear_hidden(self, batch_size):
        self.hidden_cell = (torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(self.device),
                            torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(self.device))
   
class ConvDiscriminator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.identifier = 'Medium'
        self.main = nn.Sequential(
            nn.Conv1d(input_channels,350, 20, padding =10, stride = 1),
            nn.MaxPool1d(2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv1d(350,150,10,padding=5, stride = 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv1d(150,50,6, padding = 3, stride = 1),
            nn.MaxPool1d(4,2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv1d(50, 1,4, padding = 2, stride = 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(14,output_channels)
        )

    def forward(self, input_):
        input_ = input_.permute(0,2,1)
        return self.main(input_)