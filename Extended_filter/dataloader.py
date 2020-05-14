import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from sklearn.preprocessing import MinMaxScaler

CCNSAscaler = MinMaxScaler(feature_range=(-1, 1))

def prepare_data():
    def numpy_to_tensor(tweet):
        tensor = torch.from_numpy(tweet[0:768].values)
        tweet = pd.Series([tweet['week_number'], tensor], index = ['week_number', 'encoded'])
        return tweet

    def group_by_week(tweets):
        grouped_list = []
        for name, group in tweets.groupby(['week_number']):
            batched = torch.stack(group.encoded.to_list())
            grouped_list.append(batched)
        grouped = pd.Series(grouped_list)
        return grouped

    raw_unemployment = pd.read_csv('../CCNSA.csv', sep = ';')
    unemployment_selected_dates = raw_unemployment.iloc[143:665].reset_index(drop = True)

    chunk_size = 2000
    tensored = []
    j = 0
    for chunk in pd.read_csv('train_reuters_berted.csv', dtype = np.float32, index_col = 0, chunksize = chunk_size):
        print(j)
        j+=1
        for i in chunk.index:
            tensored.append(numpy_to_tensor(chunk.loc[i]))
    tweets = pd.DataFrame(tensored)
    print('Tweets inlästa')
    tweets_by_week = group_by_week(tweets)
    print('Tweets grupperade per vecka')
    return tweets_by_week, unemployment_selected_dates
    

class DatasetTS(Dataset):
    def __init__(self, test_size, train_window, horizon):
        cls_, unemployment = prepare_data()
        self.test_size = test_size
        self.train_window = train_window
        self.horizon = horizon
        
        self.CCNSA = CCNSAscaler.fit_transform(\
            unemployment['CCNSA'].iloc[:-self.test_size].values.astype('float64') .reshape(-1, 1))[:,0]
        self.encoded = cls_ #[:-self.train_window -self.test_size] #When we load train_berted.csv there is no need to remove rows

    def __len__(self):
        return len(self.encoded) - self.train_window

    def __getitem__(self, index):
        labels = np.zeros((self.train_window))
        labels[:self.horizon] = self.CCNSA[index + self.train_window: index + self.train_window + self.horizon]
        
        return  self.CCNSA[index:index + self.train_window],\
                self.encoded[index:index + self.train_window],\
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
            enc_list.append(enc)
            label_list.append(np.expand_dims(label, 0))
        
        for i in range(0, len(seq_list), self.batch_size):
            yield (np.concatenate(seq_list[i:i + self.batch_size]),
                    enc_list[i:i + self.batch_size],
                    np.concatenate(label_list[i:i + self.batch_size]))

def get_dataloader(train_window, test_size, batch_size, horizon):
    dataset = DatasetTS(test_size, train_window, horizon)
    sampler = RandomSampler(dataset)
    dataloader = DataloaderTS(dataset, sampler, batch_size = batch_size)
    
    return dataloader
