import torch
import torch.nn as nn

class LSTMGenerator(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, horizon, \
                 init_batch_size = 1, device= torch.device('cuda:1')):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.device = device
        self.horizon = horizon

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers = num_layers, batch_first = True).to(self.device)
        self.linear = nn.Linear(hidden_layer_size, output_size).to(self.device)
        self.hidden_cell = (torch.zeros(num_layers, init_batch_size, self.hidden_layer_size).to(self.device),
                            torch.zeros(num_layers, init_batch_size, self.hidden_layer_size).to(self.device))
    
    def forward(self, input_):
        batch_z = input_.size(0)
        seq_length = input_.size(1)
        
        lstm_out, self.hidden_cell = self.lstm(input_, self.hidden_cell)
        predictions = self.linear(lstm_out.view(-1, seq_length, self.hidden_layer_size)).to(self.device)
        del lstm_out
        return predictions[:,:self.horizon,:]

    def clear_hidden(self, batch_size):
        self.hidden_cell = (torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(self.device),
                            torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(self.device))
'''
class ConvDiscriminator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.identifier = 'Large'
        self.main = nn.Sequential(
            nn.Conv1d(input_channels,1000, 20, padding =10, stride = 1),
            nn.MaxPool1d(2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv1d(1000,500,10,padding=5, stride = 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv1d(500,50,6, padding = 3, stride = 1),
            nn.MaxPool1d(4,2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv1d(50, output_channels,4, padding = 2, stride = 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(14,1)
        )

    def forward(self, input_):
        input_ = input_.permute(0,2,1)
        return self.main(input_)



class ConvDiscriminator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.identifier = 'Small'
        self.main = nn.Sequential(
            nn.Conv1d(input_channels,100, 20, padding =10, stride = 1),
            nn.MaxPool1d(2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv1d(100,20,10,padding=5, stride = 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv1d(20,5,6, padding = 3, stride = 1),
            nn.MaxPool1d(4,2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv1d(5, output_channels,4, padding = 2, stride = 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(14,1)
        )

    def forward(self, input_):
        input_ = input_.permute(0,2,1)
        return self.main(input_)

'''    
    
    
    
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
            nn.Conv1d(50, output_channels,4, padding = 2, stride = 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(14,1)
        )

    def forward(self, input_):
        input_ = input_.permute(0,2,1)
        return self.main(input_)