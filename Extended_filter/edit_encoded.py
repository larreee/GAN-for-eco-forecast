import pandas as pd
import numpy as np

chunk_size = 2000
train = []
test = []
i = 0
for chunk in pd.read_csv('../news_berted.csv', index_col = 0, dtype = np.float32, chunksize = chunk_size):
    print(i)
    i+=1
    if chunk.week_number.max() < 418:
        train.append(chunk)
    elif chunk.week_number.min() >= 418:
        test.append(chunk)
    else:
        train.append(chunk.loc[chunk.week_number < 418])
        test.append(chunk.loc[chunk.week_number >= 418])
print('Loop finished')
train_df = pd.concat(train)
test_df = pd.concat(test)
print("DF's created")
train_df.to_csv('train_news_berted.csv')
print('Train saved')
test_df.to_csv('test_news_berted.csv')
print('Test saved')
