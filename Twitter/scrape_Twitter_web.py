from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from time import sleep
import json
import datetime
import re
from random import randrange
import pandas as pd

def add_ids_to_list(ids, list_):
    for id_ in ids:
        if id_ not in list_:
            list_.append(id_)

def find_ids(driver):
    href = 'href="/\w+/status/\d+'
    ids = []
    try:
        increment = 5

        source = driver.page_source
        matches = re.findall(href, source)
        new_ids = [rege.split("/")[-1] for rege in matches]
        add_ids_to_list( new_ids, ids )

        while len(ids) >= increment:
            print('scrolling down to load more tweets')
            driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            sleep(2)
            source = driver.page_source
            matches = re.findall(href, source)
            new_ids = [rege.split("/")[-1] for rege in matches]
            
            add_ids_to_list( new_ids, ids )

            increment += 10

        print('{} tweets found'.format(len(ids)))
    except NoSuchElementException:
        print('No tweets this year')
    return ids

def format_day(date):
    day = '0' + str(date.day) if len(str(date.day)) == 1 else str(date.day)
    month = '0' + str(date.month) if len(str(date.month)) == 1 else str(date.month)
    year = str(date.year)
    return '-'.join([year, month, day])

def form_url_user(user, since, until):
    p1 = 'https://twitter.com/search?q=from%3A'
    #p2 =  user + '%20since%3A' + since + '%20until%3A' + until + 'include%3Aretweets&src=typed_query&f=live'
    p2 =  user + '%20since%3A' + since + '%20until%3A' + until + 'include%3Aretweets&src=typed_query'
    return p1 + p2

def form_url_search(since, until, search = None, exact = None, any_ = None, hashtags = None, accounts = None):
    p = 'https://twitter.com/search?q='
    search_strings = []
    if not search == None:
        search = [s.replace(' ', '%20').replace("'", "%27") for s in search]
        search_strings.append('%20OR%20'.join(search))
    if not exact == None:
        search_strings.append('"{}"'.format('%20OR%20'.join(exact)))
    if not any_ == None:
        search_strings.append('({})'.format('%20OR%20'.join(any_)))
    if not hashtags == None:
        search_strings.append('(%23{})'.format('%20OR%20%23'.join(hashtags)))
    if not accounts == None:
        search_strings.append('(from%3A{})'.format('%20OR%20from%3A'.join(accounts)))
    
    p += '%20or%20'.join(search_strings)
    p += '%20until%3A' + until+ '%20since%3A' + since + 'include%3Aretweets&src=typed_query'
    return p

def form_url_search_forecaster(search, since, until):
    search = search.replace(' ', '%20').replace("'", "%27")
    p1 = 'https://twitter.com/search?q=%22' + search
    p2 =  '%22%20(economy%20OR%20forecasting%20OR%20jobs%20OR%20economics%20OR%20econ%20OR%20fed%20OR%20prediction%20OR%20predicts%20OR%20future)%20lang%3Aen'
    p3 =  '%20until%3A' + until+ '%20since%3A' + since + 'include%3Aretweets&src=typed_query'
    return p1 + p2 + p3

def get_interval(start, end, time_increment):
    d1 = increment_day(start, 0)
    d2 = increment_day(start, time_increment)
    if d2 >= end:
        d2 = end
    return d1,d2

def increment_day(date, i):
    return date + datetime.timedelta(days=i)

def save_to_json(ids, name):
    twitter_ids_filename ='{}_all_ids.json'.format(name)
    try:
        with open(twitter_ids_filename) as f:
            all_ids = ids + json.load(f)
            data_to_write = list(set(all_ids))
            print('tweets found on this scrape: ', len(ids))
            print('total tweet count: ', len(data_to_write))
    except FileNotFoundError:
        with open(twitter_ids_filename, 'w') as f:
            all_ids = ids
            data_to_write = list(set(all_ids))
            print('tweets found on this scrape: ', len(ids))
            print('total tweet count: ', len(data_to_write))

    with open(twitter_ids_filename, 'w') as outfile:
        json.dump(data_to_write, outfile)

def scrape_from_web(start_date_input, end_date_input, id_, search = None, exact = None, any_ = None, hashtags = None, accounts = None, top_only = False):
    time_increment = 3

    start = datetime.datetime(int(start_date_input[0]), int(start_date_input[1]), int(start_date_input[2]))    # year, month, day
    end = datetime.datetime(int(end_date_input[0]), int(end_date_input[1]), int(end_date_input[2]))  # year, month, day

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--incognito")

    driver = webdriver.Chrome(chrome_options=chrome_options)

    it = 1
    ids = []

    while start < end:
        d1, d2 = get_interval(start, end, time_increment)
        url = form_url_search(format_day(d1), format_day(d2), search = search, exact = exact, any_ = any_, hashtags = hashtags, accounts = accounts)
        print(d1)
        if not top_only:
            urls = [url, url + "&f=live"]
        else:
            urls = [url]
        for u in urls:
            driver.get(u)
            sleep(2)
            if it%10 == 0:
                print("------------- 10 iterations made ----------------")
                print("------------- Time is now: {} -------------------".format(datetime.datetime.now().time()))
                print("------------- Will now restart chrome -----------")
                driver.close()
                sleep(1)
                driver = webdriver.Chrome(chrome_options=chrome_options)

            ids.extend(find_ids(driver))

            it += 1

        start = increment_day(start, time_increment)


    save_to_json(ids, id_)

    print('all done with search: {}'.format(id_))
    driver.close()

def find_years(extra):
    inputs = str(extra).split(', ')
    return inputs

###################################################################################
if __name__ == '__main__':
    reference_file = pd.ExcelFile('../references.xlsx')
    accs = pd.read_excel(reference_file, 'twitter_accounts')
    #htags = pd.read_excel(reference_file, 'twitter_hashtags')
    #columns = pd.read_excel(reference_file, 'twitter_columns')

    accs_todo = accs.loc[accs['Extra_run'].notna()]
    # searches_todo = htags.loc[htags['Web'].isna()]

    # for i in searches_todo.index:
    #     query = searches_todo.loc[i]
    #     query = query.where(query.notna(), None)
    #     try:
    #         scrape_from_web(start_date_input = [2010, 1, 1], end_date_input = [2019, 12, 31],\
    #                         id_ = query.id,\
    #                         search = [q for q in query.search.split(' ')] if query.search != None else None,\
    #                         exact = [q for q in query.exact.split(' ')] if query.exact != None else None,\
    #                         any_ = [q for q in query.any_.split(' ')] if query.any_ != None else None,\
    #                         hashtags = [q for q in query.hashtags.split(' ')] if query.hashtags != None else None,\
    #                         accounts = [q for q in query.accounts.split(' ')] if query.accounts != None else None,\
    #                         top_only = True) 
    #     except Exception as e:
    #         print("Gick knas för: {}".format(query.id))
    #         print(e)
    for i in accs_todo.index:
        try:
            scrape_from_web(start_date_input = [2009, 12, 21], end_date_input = [2019, 12, 31],\
                            id_ = accs_todo.loc[i, 'account_name'],\
                            accounts = [accs_todo.loc[i, 'account_name']]) 
        except Exception as e:
            print("Gick dåligt för: {}".format(accs_todo.loc[i, 'account_name']))
            print(e)