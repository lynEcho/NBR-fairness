import json
import sys
import os
from tqdm import tqdm
from utils.metric import evaluate
from utils.data_container import get_data_loader
from utils.load_config import get_attribute
from utils.util import convert_to_gpu
from train.train_main import create_model
from utils.util import load_model
from scipy.special import expit
import argparse
import csv
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    parser.add_argument('--fold_id', type=int, default=0, help='x')
    parser.add_argument('--best_model_path', type=str, required=True)
    
    args = parser.parse_args()

    dataset = args.dataset
    fold = args.fold_id
    model_path = args.best_model_path

    history_path = f'../../jsondata/{dataset}_history.json'
    future_path = f'../../jsondata/{dataset}_future.json'
    keyset_path = f'../../keyset/{dataset}_keyset_{fold}.json'
    

    
    with open(future_path, 'r') as f:
        data_truth = json.load(f) 


    pred_path = f'{dataset}_pred{fold}.json'
    pred_rel_path = f'{dataset}_rel{fold}.json'
    truth_path = f'{dataset}_truth{fold}.json'
    with open(keyset_path, 'r') as f:
        keyset = json.load(f)

    model = create_model()
    model = load_model(model, model_path)

    data_loader = get_data_loader(history_path=history_path,
                                    future_path=future_path,
                                    keyset_path=keyset_path,
                                    data_type='test',
                                    batch_size=1,
                                    item_embedding_matrix=model.item_embedding)

    model.eval()
    dnntsp_list = []
    
    test_key = keyset['test']
    user_ind = 0
    for step, (g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency) in enumerate(
                    tqdm(data_loader)):

        pred_data = model(g, nodes_feature, edges_weight, lengths, nodes, users_frequency) #possibility
        pred_list = pred_data.detach().squeeze(0).numpy().argsort()[::-1].tolist() #descending order
        pred_rel_list = pred_data.detach().squeeze(0).numpy().tolist() #3887users, 13897items
        truth_list = data_truth[test_key[user_ind]][1] #test_key[user_ind] is user_id


        for item in truth_list:
            dnntsp_list.append([test_key[user_ind], item, pred_rel_list[item], 1])

        i = 0
        j = 0
        while j < 100-len(truth_list):
            if pred_list[i] not in truth_list:
                dnntsp_list.append([test_key[user_ind], pred_list[i], pred_rel_list[pred_list[i]], 0])
                i = i+1 #move to next position
                j = j+1 #count remaining space
            else:
                i = i+1


        user_ind += 1

    assert len(dnntsp_list) == user_ind*100
    df = pd.DataFrame(dnntsp_list, columns=['uid', 'iid', 'score', 'label'])
    df.to_csv('/Users/lyn/Documents/code/user-fairness-main/dataset/dnntsp_rank.csv', index=False, sep=' ')


