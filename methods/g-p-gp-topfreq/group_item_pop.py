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


    sys.exit()



    data_history = pd.read_csv(f'../../dataset/{dataset}_history.csv')
    data_future = pd.read_csv(f'../../dataset/{dataset}_future.csv')

    g_top_file = f'{dataset}_pop.csv' #product_id,count(descending)
    g_top = pd.read_csv(g_top_file)
    #g_top_list = g_top['product_id'].to_list()

    '''
    #q-quantile
    purchase_counts_list = g_top['count'].to_list()
    
    print(max(purchase_counts_list))
    print(min(purchase_counts_list))

    qt = np.percentile(purchase_counts_list, [20,40,60,80]) #[13. 21. 35. 75.] divide items by q-quantile
    print(qt)
    
    '''
    


    # grouping process
    group_dict = {'pop': [], 'unpop': []}
    ind = 0
    while ind < len(g_top):

        if g_top.at[ind,'count'] >= 46:
            group_dict['pop'].append(int(g_top.at[ind,'product_id']))
        else:
            group_dict['unpop'].append(int(g_top.at[ind,'product_id']))
        ind += 1
    
    with open(f'group_results/{dataset}_group_purchase.json', 'w') as f:
        json.dump(group_dict, f)


    sys.exit()



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
    

