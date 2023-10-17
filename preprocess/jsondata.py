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
  
    data_train = pd.read_csv(f'/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/{dataset}/fold0/{dataset}_train_0.csv')
    data_valid = pd.read_csv(f'/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/{dataset}/fold0/{dataset}_valid_0.csv')
    data_test = pd.read_csv(f'/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/{dataset}/fold0/{dataset}_test_0.csv')
    
    future_dict = dict()
    history_dict = dict()
    merge_dict = dict()

    future = pd.concat([data_valid, data_test])
    merge = pd.concat([data_train, data_valid, data_test])

    #history data: {uid1: [[-1], [list], basket, ..., [-1]], uid2:[[-1], basket, basket, ..., [-1]], ... }
    data_train.sort_values(['user_id','order_number'], ascending=[True, True], inplace=True)

    for user, user_data in data_train.groupby('user_id'):
        basket_list = [[-1]]
        #print(user_data)
        #print(user)
        for order, items in user_data.groupby('order_number'):
            #print(order)
            #print(items)
            basket_list.append(list(items['item_id']))
            #print(basket_list)

        basket_list.append([-1])

        history_dict[str(user)] = basket_list

    #future data: {uid1: [[-1], basket, [-1]], uid2: [[-1], basket, [-1]], ...}
    future.sort_values(['user_id'], ascending=[True], inplace=True)

    for user, user_data in future.groupby('user_id'):

        basket = [[-1]]
        basket.append(list(user_data['item_id']))
        basket.append([-1])
        future_dict[str(user)] = basket

    #{uid1: [basket, basket, ..., basket], uid2: [basket, basket, ..., basket], ...}
    merge.sort_values(['user_id','order_number'], ascending=[True, True], inplace=True)
    for user, user_data in merge.groupby('user_id'):
        basket_list = []
        for order, items in user_data.groupby('order_number'):
            basket_list.append(list(items['item_id']))
            
        merge_dict[str(user)] = basket_list

    future_file = f'/ivi/ilps/personal/yliu10/NBR-fairness/jsondata/{dataset}_future.json'
    history_file = f'/ivi/ilps/personal/yliu10/NBR-fairness/jsondata/{dataset}_history.json'
    merge_file = f'/ivi/ilps/personal/yliu10/NBR-fairness/mergedata/{dataset}_merged.json'

    with open(future_file, 'w') as f:
        json.dump(future_dict, f)
    with open(history_file, 'w') as f:
        json.dump(history_dict, f)
    with open(merge_file, 'w') as f:
        json.dump(merge_dict, f)

