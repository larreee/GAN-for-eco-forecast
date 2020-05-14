import sys
sys.path.append("../")
from SQL_functions import sql_to_dataframe
from scrape_Twitter_web import save_to_json
import pandas as pd

def get_ids_from_sql(name):
	print('Getting ids for {}'.format(name))
	table_name = 'twitter_{}'.format(name.lower())
	df = sql_to_dataframe(table_name)
	ids = df['id_str'].tolist()
	return ids

if __name__ == '__main__':
	reference_file = pd.ExcelFile('../references.xlsx')
	searches = pd.read_excel(reference_file, 'twitter_searches')
	accs = pd.read_excel(reference_file, 'twitter_accounts')

	df = sql_to_dataframe('twitter_potus')
	print(df)
	exit()

	# accs_todo = searches['id']
	accs_todo = ['potus']
	for acc in accs_todo:
		ids = get_ids_from_sql(acc)
		save_to_json(ids, acc)