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
    args = parser.parse_args()
    dataset = args.dataset
   

    '''
    #compare the interaction of pop1 and pop2
    purchase_file = f'/ivi/ilps/personal/yliu10/NBR-fairness/methods/g-p-gp-topfreq/group_results/{dataset}_group_purchase.json'
    user_file = f'/ivi/ilps/personal/yliu10/NBR-fairness/methods/g-p-gp-topfreq/group_results/{dataset}_group_user.json'

    with open(purchase_file, 'r') as f:
        purchase = json.load(f)
    with open(user_file, 'r') as f:
        user = json.load(f)
    
    pop = list(set(purchase["pop"]).intersection(user["pop"]))
    unpop = list(set(purchase["unpop"]).intersection(user["unpop"]))

    print(len(purchase["pop"]))
    print(len(user["pop"]))
    print(len(pop))
    print(len(pop) / len(purchase["pop"]))
    print(len(pop) / len(user["pop"]))
    
    print(len(purchase["unpop"]))
    print(len(user["unpop"]))
    print(len(unpop))
    print(len(unpop) / len(purchase["unpop"]))
    print(len(unpop) / len(user["unpop"]))
    '''
    '''
    data_history = pd.read_csv(f'/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/{dataset}/fold/{dataset}_train.csv')
    data_valid = pd.read_csv(f'/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/{dataset}/fold/{dataset}_valid.csv')
    data_test = pd.read_csv(f'/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/{dataset}/fold/{dataset}_test.csv')

    all_data = pd.concat([data_history, data_valid, data_test], ignore_index=True)
    
    #generate pop.csv
    item_counts = all_data['item_id'].value_counts()
    item_counts_df = item_counts.reset_index()
    item_counts_df.columns = ['item_id', 'count']
    item_counts_df.to_csv(f'/ivi/ilps/personal/yliu10/NBR-fairness/methods/g-p-gp-topfreq/pop/{dataset}_pop_all.csv', index=False)
    '''

    g_top_file = f'pop/{dataset}_pop.csv' #item_id,count(descending)
    g_top = pd.read_csv(g_top_file)

    '''
    #q-quantile
    purchase_counts_list = g_top['count'].to_list()
    
    print(max(purchase_counts_list))
    print(min(purchase_counts_list))

    qt = np.percentile(purchase_counts_list, [10,20,30,40,50,60,70,80,90]) #divide items by q-quantile
    print(qt)
    
    '''
    if dataset == 'instacart':
        threshold = 87
    elif dataset == 'dunnhumby':
        threshold = 33
    elif dataset == 'tafeng':
        threshold = 42

    # grouping process
    group_dict = {'pop': [], 'unpop': []}
    ind = 0
    while ind < len(g_top):

        if g_top.at[ind,'count'] > threshold:
            group_dict['pop'].append(int(g_top.at[ind,'item_id']))
        else:
            group_dict['unpop'].append(int(g_top.at[ind,'item_id']))
        ind += 1
    
    with open(f'group_results/{dataset}_group_purchase.json', 'w') as f:
        json.dump(group_dict, f)

    

    '''
    #pop1 = number of purchases

    product_counts_list = g_top['count'].to_list()
    products_counts_counter = Counter(product_counts_list)
    
    #plt.subplot(1, 2, 1)
    plt.plot(products_counts_counter.keys(), products_counts_counter.values())
    plt.xscale('log')
    plt.xlabel('number of purchases (log)')
    plt.ylabel('number of items')
    plt.show()
    '''


    '''

    prod_user = data_history[['product_id','user_id']].drop_duplicates()
    #print(prod_user.head())
    group_product = prod_user.groupby('product_id', as_index=False).agg({'user_id':'count'}) #product_id, how many users bought it
    
    # grouping process
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


    '''

    '''
    #q-quantile
    peo_counts_list = group_product['user_id'].to_list()
    
    print(max(peo_counts_list))
    print(min(peo_counts_list))
    qt = np.percentile(peo_counts_list, [50]) #[13. 21. 35. 75.] divide items by q-quantile
    print(qt)
    
    '''
    
    '''
    pop2 = number of users who purchased this item
    peo_counts_counter = Counter(peo_counts_list)

    plt.subplot(1, 2, 2)
    plt.plot(peo_counts_counter.keys(), peo_counts_counter.values())
    plt.xscale('log')
    plt.xlabel('number of users (log)')
    plt.ylabel('number of items')
    plt.show()
    '''
    

