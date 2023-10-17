import pandas as pd
import json
import random
import argparse
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    parser.add_argument('--fold_id', type=int, default=0, help='x')
    args = parser.parse_args()
    dataset = args.dataset
    fold_id = args.fold_id
    
    data_train = pd.read_csv(f'/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/{dataset}/fold0/{dataset}_train_0.csv')
    data_valid = pd.read_csv(f'/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/{dataset}/fold0/{dataset}_valid_0.csv')
    data_test = pd.read_csv(f'/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/{dataset}/fold0/{dataset}_test_0.csv')

    future = pd.concat([data_valid, data_test])
    merge = pd.concat([data_train, data_valid, data_test])
    if fold_id == 0:
        
        train_user = [str(element) for element in list(set(data_train['user_id']))]
        val_user = [str(element) for element in list(set(data_valid['user_id']))]
        test_user = [str(element) for element in list(set(data_test['user_id']))]
    else:
        user = list(set(future['user_id']))
        user_num = len(user)
        random.shuffle(user)
        user = [str(user_id) for user_id in user]
        
        train_user = user
        val_user = user[:int(user_num*1/2)]
        test_user = user[int(user_num*1/2):]
    
    item_num = len(set(merge['item_id']))
    keyset_dict = dict()
    keyset_dict['item_num'] = item_num+1 #all items
    keyset_dict['train'] = train_user
    keyset_dict['val'] = val_user
    keyset_dict['test'] = test_user

    if not os.path.exists('/ivi/ilps/personal/yliu10/NBR-fairness/keyset/'):
        os.makedirs('/ivi/ilps/personal/yliu10/NBR-fairness/keyset/')
    keyset_file = f'/ivi/ilps/personal/yliu10/NBR-fairness/keyset/{dataset}_keyset_{fold_id}.json'
    with open(keyset_file, 'w') as f:
        json.dump(keyset_dict, f)


