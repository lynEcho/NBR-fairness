import pandas as pd
import json
import random
import argparse
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    
    args = parser.parse_args()
    dataset = args.dataset
    
    
    data_train = pd.read_csv(f'../csvdata/{dataset}/{dataset}_train.csv')
    data_valid = pd.read_csv(f'../csvdata/{dataset}/{dataset}_valid.csv')
    data_test = pd.read_csv(f'../csvdata/{dataset}/{dataset}_test.csv')

    future = pd.concat([data_valid, data_test])
    merge = pd.concat([data_train, data_valid, data_test])
   
    
    train_user = [str(element) for element in list(set(data_train['user_id']))]
    val_user = [str(element) for element in list(set(data_valid['user_id']))]
    test_user = [str(element) for element in list(set(data_test['user_id']))]


    item_num = len(set(merge['item_id']))
    keyset_dict = dict()
    keyset_dict['item_num'] = item_num+1 #all items
    keyset_dict['train'] = train_user
    keyset_dict['val'] = val_user
    keyset_dict['test'] = test_user

    if not os.path.exists('../keyset/'):
        os.makedirs('../keyset/')
    keyset_file = f'../keyset/{dataset}_keyset.json'
    with open(keyset_file, 'w') as f:
        json.dump(keyset_dict, f)


