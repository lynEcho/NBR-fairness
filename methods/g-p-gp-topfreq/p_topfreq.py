import pandas as pd
import json
import argparse
import os
import sys
from scipy.special import expit
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    parser.add_argument('--fold_id', type=int, default=0, help='x')
    args = parser.parse_args()
    dataset = args.dataset
    fold_id = args.fold_id
    data_history = pd.read_csv(f'../../dataset/{dataset}_history.csv')
    data_future = pd.read_csv(f'../../dataset/{dataset}_future.csv')

    keyset_file = f'/ivi/ilps/personal/yliu10/NBR-fairness/keyset/{dataset}_keyset_{fold_id}.json'   
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)

    pred_all_dict = dict()
    pred_dict = dict() #pred for test users
    pred_rel_all_dict = dict()
    pred_rel_dict = dict() #relevance for test users

    for user, user_data in data_future.groupby('user_id'):
        
        user_history = data_history[data_history['user_id'].isin([user])]
       
        history_items = user_history['product_id'].tolist()
        
        # print(history_items)
        s_pop_dict = dict()
        for item in history_items:
            if item not in s_pop_dict.keys():
                s_pop_dict[item] = 1
            else:
                s_pop_dict[item] += 1
        s_dict = sorted(s_pop_dict.items(), key=lambda d: d[1], reverse=True) #[(0, 10), (4, 10), (6, 9), (8, 8)...]
        
        pred = []
        rel = [0] * 3920 #instacart = 13897, dunnhumby = 3920, tafeng = 11997
        for item, cnt in s_dict:
            pred.append(item)
            rel[item] = cnt

        pred_all_dict[user] = pred
        pred_rel_all_dict[user] = expit(rel).tolist() 
    
    
    for user in keyset['test']: 
        pred_dict[user] = pred_all_dict[int(user)]
        pred_rel_dict[user] = pred_rel_all_dict[int(user)]


    if not os.path.exists('p_top_pred_testu/'):
        os.makedirs('p_top_pred_testu/')
    with open(f'p_top_pred_testu/{dataset}_pred{fold_id}.json', 'w') as f:
        json.dump(pred_dict, f)

    with open(f'p_top_pred_testu/{dataset}_rel{fold_id}.json', 'w') as f:
        json.dump(pred_rel_dict, f)



