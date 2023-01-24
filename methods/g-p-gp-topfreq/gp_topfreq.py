import pandas as pd
import json
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
    data_history = pd.read_csv(f'../../dataset/{dataset}_history.csv')
    data_future = pd.read_csv(f'../../dataset/{dataset}_future.csv')

    g_top_file = f'{dataset}_pop.csv' #product_id,count
    g_top = pd.read_csv(g_top_file)
    g_top_list = g_top['product_id'].to_list()
    
    data_history = pd.read_csv(f'../../dataset/{dataset}_history.csv')
    data_future = pd.read_csv(f'../../dataset/{dataset}_future.csv')
   
    pred_dict = dict()
    for user, user_data in data_future.groupby('user_id'):

        user_history = data_history[data_history['user_id'].isin([user])] #one user every time
        history_items = user_history['product_id'].tolist()
        s_pop_dict = dict() #{item: count}
        for item in history_items:
            if item not in s_pop_dict.keys():
                s_pop_dict[item] = 1
            else:
                s_pop_dict[item] += 1
        s_dict = sorted(s_pop_dict.items(), key=lambda d: d[1], reverse=True) #sort items via decreasing count, list [(0, 10), (4, 10), (6, 9), (8, 8), ...]
        
        pred = []
        for item, cnt in s_dict:
            pred.append(item)

        ind = 0
        while(len(pred)<100):
            if g_top_list[ind] not in pred:
                pred.append(g_top_list[ind])
            ind += 1
        pred_dict[user] = pred
        
    if not os.path.exists('gp_top_pred/'):
        os.makedirs('gp_top_pred/')
    with open(f'gp_top_pred/{dataset}_pred{fold_id}.json', 'w') as f:
        json.dump(pred_dict, f)



