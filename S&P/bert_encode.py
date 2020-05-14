from transformers import DistilBertModel
import pandas as pd
import torch

model = DistilBertModel.from_pretrained('distilbert-base-uncased')

if torch.cuda.is_available():
	device = torch.device('0')
	torch.cuda.set_device(device = 0)
	LT = torch.cuda.LongTensor
	model.to(device)
else:
	LT = torch.LongTensor

def encode(toks):
	tok_tensor = torch.from_numpy(toks.tokens).type(LT)
	with torch.no_grad():
		out = model(tok_tensor)
	return pd.Series([toks.week_number, out[0][:,0,:].numpy()], index = ['week_number','encoded'])


df = pd.read_hdf('full_tokenized_batched_1000.h5')
print('Data read')
bert_list = []
for i in range(len(df.index)):
	print('Starting week {}'.format(i))
	df_temp = df.iloc[i].apply(encode, axis = 1)
	# df_temp['week_number'] = pd.Series(i, index = df_temp.index)
	bert_list.append(df_temp)

full = pd.concat(bert_list)
final = pd.Series([group for name, group in full.groupby(['week_number'])],\
                                                name = 'encoded_group')
final.to_hdf('full_berted.h5','berted')
# final.to_csv('full_berted.csv')