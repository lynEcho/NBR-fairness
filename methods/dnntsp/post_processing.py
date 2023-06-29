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
    args = parser.parse_args()
    dataset = args.dataset
    fold = args.fold_id
    
    
    pred_rel_path = f'/Users/lyn/Documents/code/NBR-fairness/methods/dnntsp/{dataset}_rel{fold}.json'
    with open(pred_rel_path, 'r') as f:
        pred_rel_dict = json.load(f)


    pred_rel_list = []
    for i in pred_rel_dict.values():
        pred_rel_list.append(i)
    
    #3887users, 13897items
    
    bounds = [[0,1]] * len(pred_rel_list) 
    
    fair_list = ILPBased(np.array(pred_rel_list), np.array(bounds))
    fair_list.prepare(100, 20, 0.5, 0.99)
    
    fair_list.start(1)
    fair_list.get_result

    fair_dict = dict()
    fair_rel_dict = dict()

    j = 0
    for i in pred_rel_dict.keys():
        fair_dict[i] = fair_list.final[j].tolist()
        fair_rel_dict[i] = fair_list.rerank_rel[j]
        j = j+1


    fair_path = f'{dataset}_fair1{fold}.json'
    fair_rel_path = f'{dataset}_fair1_rel{fold}.json'

    with open(fair_path, 'w') as f:
        json.dump(fair_dict, f)
    with open(fair_rel_path, 'w') as f:
        json.dump(fair_rel_dict, f)

