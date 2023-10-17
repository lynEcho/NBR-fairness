import pandas as pd
import json
import argparse
import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import random
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

    g_top_file = f'{dataset}_pop.csv' #product_id,count(descending)
    g_top = pd.read_csv(g_top_file)
    g_top_list = g_top['product_id'].to_list()

    ''' 
    #pop1 = number of purchases

    product_counts_list = g_top['count'].to_list()
    products_counts_counter = Counter(product_counts_list)
    
    plt.subplot(1, 2, 1)
    plt.plot(products_counts_counter.keys(), products_counts_counter.values())
    plt.xscale('log')
    plt.xlabel('number of purchases (log)')
    plt.ylabel('number of items')
    '''

    prod_user = data_history[['product_id','user_id']].drop_duplicates()
    #print(prod_user.head())
    group_product = prod_user.groupby('product_id', as_index=False).agg({'user_id':'count'})
    
    prod_peonum = group_product.sort_values('user_id', ascending=False) #product_id, how many users bought it
    
    group_dict = {'pop': [], 'unpop': []}
    
    ind = 0
    while ind < len(group_product):

        if group_product.at[ind,'user_id'] >= 306:
            group_dict['pop'].append(int(group_product.at[ind,'product_id']))
        else:
            group_dict['unpop'].append(int(group_product.at[ind,'product_id']))
        ind += 1
    
    with open(f'group_results/{dataset}_group_user.json', 'w') as f:
        json.dump(group_dict, f)







    sys.exit()

    '''
    for row in prod_peonum.iterrows():
        if row[1]['user_id'] <= 20:
            group_dict['group0'].append(int(row[1]['product_id']))
        elif row[1]['user_id'] >= 21 and row[1]['user_id'] <= 74:
            group_dict['group1'].append(int(row[1]['product_id']))
        elif row[1]['user_id'] >= 75:
            group_dict['group2'].append(int(row[1]['product_id']))

    with open(f'gp_top_pred/{dataset}_group_des.json', 'w') as f:
        json.dump(group_dict, f)
    
    '''
    '''
    peo_counts_list = group_product['user_id'].to_list()
    
    print(max(peo_counts_list))
    print(min(peo_counts_list))
    qt = np.percentile(peo_counts_list, [50]) #[13. 21. 35. 75.] divide items by q-quantile
    print(qt)
    
    '''
    
    '''
    peo_counts_counter = Counter(peo_counts_list)

    plt.subplot(1, 2, 2)
    plt.plot(peo_counts_counter.keys(), peo_counts_counter.values())
    plt.xscale('log')
    plt.xlabel('number of users (log)')
    plt.ylabel('number of items')
    plt.show()
    '''
    
    pred_dict = dict()
    item_count_dict = dict()
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
    
        item_count_dict[user] = s_pop_dict
        pred = []
        for item, cnt in s_dict:
            pred.append(item)


        '''
        g_top_part = g_top_list[:100] #top100
        random.shuffle(g_top_part)
        
        g_top_part = g_top_list #top
        random.shuffle(g_top_part)
        
        #g_top_part = g_top_list #GP-TopFreq
        
        ind = 0
        while(len(pred)<100):
            if g_top_part[ind] not in pred:
                pred.append(g_top_part[ind])
            ind += 1
        pred_dict[user] = pred
        
        '''

        
        init = len(pred) #length of repeat items
        if init < 20: #fill in the expl
            expl = 20 - len(pred)
            if expl % 3 == 0:
                ind = 0
                while(len(pred) < init+(expl//3)):
                    if group_dict['group2'][ind] not in pred:
                        pred.append(group_dict['group2'][ind])
                    ind += 1
                
                ind = 0
                while(len(pred) < init+2*(expl//3)):
                    if group_dict['group1'][ind] not in pred:
                        pred.append(group_dict['group1'][ind])
                    ind += 1
                
                ind = 0
                while(len(pred) < init+3*(expl//3)):
                    if group_dict['group0'][ind] not in pred:
                        pred.append(group_dict['group0'][ind])
                    ind += 1

            elif expl % 3 == 1:
                ind = 0
                while(len(pred) < init+(expl//3)+1):
                    if group_dict['group2'][ind] not in pred:
                        pred.append(group_dict['group2'][ind])
                    ind += 1
                
                ind = 0
                while(len(pred) < init+2*(expl//3)+1):
                    if group_dict['group1'][ind] not in pred:
                        pred.append(group_dict['group1'][ind])
                    ind += 1
                
                ind = 0
                while(len(pred) < init+3*(expl//3)+1):
                    if group_dict['group0'][ind] not in pred:
                        pred.append(group_dict['group0'][ind])
                    ind += 1

            elif expl % 3 == 2:
                ind = 0
                while(len(pred) < init+(expl//3)+1):
                    if group_dict['group2'][ind] not in pred:
                        pred.append(group_dict['group2'][ind])
                    ind += 1
                
                ind = 0
                while(len(pred) < init+2*(expl//3)+2):
                    if group_dict['group1'][ind] not in pred:
                        pred.append(group_dict['group1'][ind])
                    ind += 1
                
                ind = 0
                while(len(pred) < init+3*(expl//3)+2):
                    if group_dict['group0'][ind] not in pred:
                        pred.append(group_dict['group0'][ind])
                    ind += 1
                
        assert len(pred) >= 20
        pred_dict[user] = pred
        



    if not os.path.exists('gp_top_pred/'):
        os.makedirs('gp_top_pred/')
    with open(f'gp_top_pred/{dataset}_pred{fold_id}_ave20.json', 'w') as f:
        json.dump(pred_dict, f)
    '''
    with open(f'gp_top_pred/{dataset}_group.json', 'w') as f:
        json.dump(group_dict, f)

    with open(f'gp_top_pred/{dataset}_group_des.json', 'w') as f:
        json.dump(group_dict, f)
    
    with open(f'gp_top_pred/{dataset}_item_count.json', 'w') as f:
        json.dump(item_count_dict, f)
    '''