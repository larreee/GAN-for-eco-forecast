import torch
import torch.nn as nn


def noise(size, FT):
    '''
    Generates a vector of gaussian sampled random values
    '''
    with torch.no_grad():
        n = torch.randn(size).type(FT)
        return n
    
def cat_with_seq_with_enc(input_1, seq, encoded):
    return torch.cat((input_1, seq, encoded),2)

def cat_no_seq_with_enc(input_1, seq, encoded):
    return torch.cat((input_1, encoded),2)

def cat_with_seq_no_enc(input_1, seq, encoded):
    return torch.cat((input_1, seq),2)

def cat_no_seq_no_enc(input_1, seq, encoded):
    return input_1

def pad(tensor, train_window, horizon):
    return nn.functional.pad(tensor, (0,0,0, train_window - horizon), mode = 'constant', value = 0)