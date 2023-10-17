import pandas as pd
import json
import argparse
import os
import sys
#from scipy.special import expit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    parser.add_argument('--fold_id', type=int, default=0, help='x')
    args = parser.parse_args()
    dataset = args.dataset
    fold_id = args.fold_id

    data_history = pd.read_csv(f'/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/{dataset}/fold/{dataset}_train.csv')
    
    keyset_file = f'/ivi/ilps/personal/yliu10/NBR-fairness/keyset/{dataset}_keyset_{fold_id}.json'   
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)

    '''
    #generate pop.csv
    item_counts = data_history['item_id'].value_counts()
    item_counts_df = item_counts.reset_index()
    item_counts_df.columns = ['item_id', 'count']
    item_counts_df.to_csv(f'/ivi/ilps/personal/yliu10/NBR-fairness/methods/g-p-gp-topfreq/pop/{dataset}_pop.csv', index=False)
    '''

    g_top_file = f'pop/{dataset}_pop.csv'
    g_top = pd.read_csv(g_top_file)

    top_items = g_top.head(100)
    gtop_dict = dict(zip(top_items['item_id'], top_items['count']))


    pred_dict = dict()
    rel_dict = dict()

    #same rec. for each user

    pred = []
    rel = [0] * keyset['item_num']
    for item, cnt in gtop_dict.items():
        pred.append(item)
        rel[item] = cnt
    
    for user in keyset['test']:
        pred_dict[user] = pred
        max_rel = max(rel)
        rel_dict[user] = [x / max_rel for x in rel] #divided by the max value

        #rel_dict[user] = expit(rel).tolist() 


    if not os.path.exists('g_top_results/'):
        os.makedirs('g_top_results/')

    with open(f'g_top_results/{dataset}_pred{fold_id}.json', 'w') as f:
        json.dump(pred_dict, f)
    with open(f'g_top_results/{dataset}_rel{fold_id}.json', 'w') as f:
        json.dump(rel_dict, f)


