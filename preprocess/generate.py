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
    
    data_valid = pd.read_csv(f'/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/{dataset}/fold0/{dataset}_valid_0.csv')
    data_test = pd.read_csv(f'/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/{dataset}/fold0/{dataset}_test_0.csv')

    valid_baskets_file_path = f'/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/{dataset}/fold{fold_id}/{dataset}_valid_{fold_id}.csv'
    test_baskets_file_path = f'/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/{dataset}/fold{fold_id}/{dataset}_test_{fold_id}.csv'

    keyset_path = f'/ivi/ilps/personal/yliu10/NBR-fairness/keyset/{dataset}_keyset_{fold_id}.json'
    with open(keyset_path, 'r') as json_file:
        keyset = json.load(json_file)

    future = pd.concat([data_valid, data_test])
    
    valid_df = future[future['user_id'].isin(keyset['val'])]
    test_df = future[future['user_id'].isin(keyset['test'])]

    assert len(set(valid_df['user_id'])) == len(set(data_valid['user_id']))
    assert len(set(test_df['user_id'])) == len(set(data_test['user_id']))

    valid_df.to_csv(valid_baskets_file_path,index=False)
    test_df.to_csv(test_baskets_file_path,index=False)

