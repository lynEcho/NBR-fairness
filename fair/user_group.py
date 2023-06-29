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
    data_history = pd.read_csv(f'/Users/lyn/Documents/code/NBR-fairness/dataset/{dataset}_history.csv')
    data_future = pd.read_csv(f'/Users/lyn/Documents/code/NBR-fairness/dataset/{dataset}_future.csv')

    keyset_file = f'/Users/lyn/Documents/code/NBR-fairness/keyset/{dataset}_keyset_{fold_id}.json'   
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)

    history_file = f'/Users/lyn/Documents/code/NBR-fairness/jsondata/{dataset}_history.json'
    with open(history_file, 'r') as f:
        history = json.load(f)


    repeat = ['uid']
    explore = ['uid']
    '''
    repeat_ratio_dict = dict()

    for user in keyset['test']:
        
        basket_list = history[user][1:-1]
        history_set = set(basket_list[0])
        repeat_ratio = []

        for i in range(1, len(basket_list)): 
            current_basket = set(basket_list[i])
            repeat_ratio.append(len(history_set&current_basket) / len(current_basket))
            history_set = history_set|current_basket

        repeat_ratio_dict[user] = repeat_ratio

        if sum(repeat_ratio[-3:])/len(repeat_ratio[-3:]) >= 0.5:
        #if repeat_ratio[-1] >= 0.5:
            repeat.append(user)
        else:
            explore.append(user)

    
    '''
    for user, user_data in data_future.groupby('user_id'):
        
        if str(user) in keyset['test']: 
            
            user_history = data_history[data_history['user_id'].isin([user])]
        
            history_items = user_history['product_id'].tolist()
            
            future_items = user_data['product_id'].tolist()

            if len(set(history_items)&set(future_items)) / len(set(future_items)) >= 0.5:
                repeat.append(str(user))
            else:
                explore.append(str(user))
    
    print(len(repeat)-1)
    print(len(explore)-1)
    
    with open(f'/Users/lyn/Documents/code/user-fairness-main/dataset/{dataset}_repeat.txt', 'w') as output:
        output.write('\n'.join(repeat))
    with open(f'/Users/lyn/Documents/code/user-fairness-main/dataset/{dataset}_explore.txt', 'w') as output:
        output.write('\n'.join(explore))


