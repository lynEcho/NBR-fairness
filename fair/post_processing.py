import json
import sys
import os
from tqdm import tqdm

from ilp.ilpbased import ILPBased
import argparse
from numpy.core.fromnumeric import ndim, shape, argsort
import numpy as np
from ilp.extra_functions import pos_bias
#clean the package


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    parser.add_argument('--fold_id', type=int, default=0, help='x')
    parser.add_argument('--rel_folder_path', type=str, default='/Users/lyn/Documents/code/NBR-fairness/methods/g-p-gp-topfreq/p_top_pred', help='rel_folder')
    args = parser.parse_args()
    dataset = args.dataset
    fold = args.fold_id
    rel_folder = args.rel_folder_path
    
    pred_rel_path = f'{rel_folder}/{dataset}_rel_test{fold}.json'
    with open(pred_rel_path, 'r') as f:
        pred_rel_dict = json.load(f)


    pred_rel_list = []
    for i in pred_rel_dict.values():
        pred_rel_list.append(i)
    
    #3887users, 13897items
    
    bounds = [[0,1]] * len(pred_rel_list) 
    
    fair_list = ILPBased(np.array(pred_rel_list), np.array(bounds))
    fair_list.prepare(100, 20, 0.5, 0.99)
    
    fair_list.start(2)
    fair_list.get_result
    
    fair_dict = dict()
    fair_rel_dict = dict()

    j = 0
    for i in pred_rel_dict.keys():
        fair_dict[i] = fair_list.final[j].tolist()
        fair_rel_dict[i] = fair_list.rerank_rel[j]
        j = j+1


    fair_path = f'{rel_folder}/{dataset}_fair{fold}.json'
    fair_rel_path = f'{rel_folder}/{dataset}_fair_rel{fold}.json'

    with open(fair_path, 'w') as f:
        json.dump(fair_dict, f)
    with open(fair_rel_path, 'w') as f:
        json.dump(fair_rel_dict, f)

