import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from sklearn.preprocessing import MinMaxScaler

CCNSAscaler = MinMaxScaler(feature_range=(-1, 1))

def prepare_data():
    raw_unemployment = pd.read_csv('../CCNSA.csv', sep = ';')
    unemployment_selected_dates = raw_unemployment.iloc[143:665].reset_index(drop = True)

    tweets = pd.read_csv('../Berted_by_week.csv', index_col = 0)
    cleaned_tweets = []
    for element in tweets['1']:
        #print(element.replace("\n",'').strip('tensor').strip("([").strip("])").replace(' ','').split(','))
        cleaned_tweets.append(element.replace("\n",'').strip('tensor').strip("([").strip("])").replace(' ','').split(','))

    cls_ = []
    for j in cleaned_tweets:
        tmp = []
        for i in j:
            tmp.append(float(i))
        cls_.append(tmp)
    #print(type(cls[0][0]))
    cls_array = np.array(cls_)
    
    return cls_array, unemployment_selected_dates
    

class DatasetTS(Dataset):
    def __init__(self, test_size, train_window, horizon):
        cls, unemployment = prepare_data()
        self.test_size = test_size
        self.train_window = train_window
        self.horizon = horizon
        
        self.CCNSA = CCNSAscaler.fit_transform(\
            unemployment['CCNSA'].iloc[:-self.test_size].values.astype('float64') .reshape(-1, 1))[:,0]
        self.text = cls[:-self.train_window -self.test_size,:]
    def __len__(self):
        return 522 - 2*self.train_window - self.test_size

    def __getitem__(self, index):
        labels = np.zeros((self.test_size))
        labels[:self.horizon] = self.CCNSA[index + self.train_window: index + self.train_window + self.horizon]
        
        return self.CCNSA[index:index + self.train_window],\
                self.text[index:index + self.train_window,:],\
                labels
                

class DataloaderTS():
    def __init__(self, Dataset, Sampler, batch_size):
        self.dataset = Dataset
        self.sampler = Sampler
        self.batch_size = batch_size
        
    def __iter__(self):
        seq_list = []
        enc_list = []
        label_list = []
        
        for sample in self.sampler:
            seq, enc, label = self.dataset[sample]
            seq_list.append(np.expand_dims(seq, 0))
            enc_list.append(np.expand_dims(enc,0))
            label_list.append(np.expand_dims(label, 0))
        
        for i in range(0, len(seq_list), self.batch_size):
            yield (np.concatenate(seq_list[i:i + self.batch_size]),
                    np.concatenate(enc_list[i:i + self.batch_size]),
                    np.concatenate(label_list[i:i + self.batch_size]))

def get_dataloader(train_window, test_size, batch_size, horizon):
    dataset = DatasetTS(train_window, test_size, horizon)
    sampler = RandomSampler(dataset, replacement = True)
    dataloader = DataloaderTS(dataset, sampler, batch_size = batch_size)
    
    return dataloader
