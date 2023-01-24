from metrics import *
import pandas as pd
import json
import argparse
import os
import sys


def get_repeat_eval(pred_folder, dataset, size, fold_list, file):
    history_file = f'../dataset/{dataset}_history.csv'
    truth_file = f'../jsondata/{dataset}_future.json'
    data_history = pd.read_csv(history_file)
    with open(truth_file, 'r') as f:
        data_truth = json.load(f) #dict {'19564': [[-1], [7690, 156, 154, 481, 6876, 2315, 2665, 1286, 1944, 75, 206, 20, 294, 18, 1880, 259, 188, 407, 3745], [-1]]} 

    group_file = f'{pred_folder}/{dataset}_group.json' #note here
    with open(group_file, 'r') as f:
        item_group = json.load(f)

    '''    
    item_count_file = f'{pred_folder}/{dataset}_item_count.json'
    with open(item_count_file, 'r') as f:
        item_count = json.load(f)
    '''
  
    a_ndcg = []
    a_recall = []
    a_hit = []
    a_repeat_ratio = []
    a_explore_ratio = []
    a_recall_repeat = []
    a_recall_explore = []
    a_hit_repeat = []
    a_hit_explore = []
    a_fairness = []
    a_fairness_rep = []
    a_fairness_expl = []

    for ind in fold_list:
        keyset_file = f'../keyset/{dataset}_keyset_{ind}.json'
        pred_file = f'{pred_folder}/{dataset}_pred{ind}.json' #change here
        with open(keyset_file, 'r') as f:
            keyset = json.load(f)
        with open(pred_file, 'r') as f:
            data_pred = json.load(f)
        
        # compute fold
        ndcg = []
        recall = []
        hit = []
        repeat_ratio = []
        explore_ratio = []
        recall_repeat = []
        recall_explore = []
        hit_repeat = []
        hit_explore = []
        exp0 = []
        exp1 = []
        exp2 = []
        u0 = []
        u1 = []
        u2 = []
        exp0_rep = []
        exp1_rep = []
        exp2_rep = []
        exp0_expl = []
        exp1_expl = []
        exp2_expl = []
        u0_rep = []
        u1_rep = []
        u2_rep = []
        u0_expl = []
        u1_expl = []
        u2_expl = []

        for user in keyset['test']: #only test users are evaluated
            pred = data_pred[user][:size]
            truth = data_truth[user][1]
            #print(user)
            user_history = data_history[data_history['user_id'].isin([int(user)])] #user_id  order_number  product_id
            repeat_items = list(set(user_history['product_id']))
            truth_repeat = list(set(truth)&set(repeat_items)) # might be none, repeat items in ground truth
            truth_explore = list(set(truth)-set(truth_repeat)) # might be none, explore items in ground truth

            pred_repeat = list(set(pred)&set(repeat_items)) #repeat items in pred, might be none
            pred_explore = list(set(pred)-set(pred_repeat)) #explore items in pred, might be none


            u_exp0, u_exp1, u_exp2 = get_Exposure(pred, item_group, size)
            exp0.append(u_exp0)
            exp1.append(u_exp1)
            exp2.append(u_exp2)

            u_u0, u_u1, u_u2 = get_Utility(item_group, truth, pred, size)
            
            u0.append(u_u0)
            u1.append(u_u1)
            u2.append(u_u2)

            u_ndcg = get_NDCG(truth, pred, size)
            ndcg.append(u_ndcg)
            u_recall = get_Recall(truth, pred, size)
            recall.append(u_recall)
            u_hit = get_HT(truth, pred, size)
            hit.append(u_hit)

            u_repeat_ratio, u_explore_ratio = get_repeat_explore(repeat_items, pred, size)# here repeat items
            repeat_ratio.append(u_repeat_ratio)
            explore_ratio.append(u_explore_ratio)

            if len(truth_repeat)>0:
                u_recall_repeat = get_Recall(truth_repeat, pred, size)# here repeat truth, since repeat items might not in the groundtruth
                recall_repeat.append(u_recall_repeat)
                u_hit_repeat = get_HT(truth_repeat, pred, size)
                hit_repeat.append(u_hit_repeat)

            if len(truth_explore)>0:
                u_recall_explore = get_Recall(truth_explore, pred, size)
                u_hit_explore = get_HT(truth_explore, pred, size)
                recall_explore.append(u_recall_explore)
                hit_explore.append(u_hit_explore)
        
            if len(pred_repeat)>0:
                u_exp0_rep, u_exp1_rep, u_exp2_rep = get_Exposure(pred_repeat, item_group, size)
            else:
                u_exp0_rep = u_exp1_rep = u_exp2_rep = 0
            exp0_rep.append(u_exp0_rep)
            exp1_rep.append(u_exp1_rep)
            exp2_rep.append(u_exp2_rep)


            if len(pred_explore)>0:    
                u_exp0_expl, u_exp1_expl, u_exp2_expl = get_Exposure(pred_explore, item_group, size)
            else:
                u_exp0_expl = u_exp1_expl = u_exp2_expl = 0
            exp0_expl.append(u_exp0_expl)
            exp1_expl.append(u_exp1_expl)
            exp2_expl.append(u_exp2_expl)



            if len(truth_repeat)>0 and len(pred_repeat)>0:
                u_u0_rep, u_u1_rep, u_u2_rep = get_Utility(item_group, truth_repeat, pred_repeat, size)
            else:
                u_u0_rep = u_u1_rep = u_u2_rep = 0
            u0_rep.append(u_u0_rep)
            u1_rep.append(u_u1_rep)
            u2_rep.append(u_u2_rep)

            if len(truth_explore)>0 and len(pred_explore)>0:
                u_u0_expl, u_u1_expl, u_u2_expl = get_Utility(item_group, truth_explore, pred_explore, size)
            else:
                u_u0_expl = u_u1_expl = u_u2_expl = 0
            u0_expl.append(u_u0_expl)
            u1_expl.append(u_u1_expl)
            u2_expl.append(u_u2_expl)
            
            assert u_exp0_rep + u_exp0_expl == u_exp0
            assert u_exp1_rep + u_exp1_expl == u_exp1
            assert u_exp2_rep + u_exp2_expl == u_exp2

            assert u_u0_rep + u_u0_expl == u_u0
            assert u_u1_rep + u_u1_expl == u_u1
            assert u_u2_rep + u_u2_expl == u_u2

        print('exp0:', np.sum(exp0))
        print('exp1:', np.sum(exp1))
        print('exp2:', np.sum(exp2))

        print('exp0_rep:', np.sum(exp0_rep))
        print('exp1_rep:', np.sum(exp1_rep))
        print('exp2_rep:', np.sum(exp2_rep))

        print('exp0_expl:', np.sum(exp0_expl))
        print('exp1_expl:', np.sum(exp1_expl))
        print('exp2_expl:', np.sum(exp2_expl))

        print('u0:', np.sum(u0))
        print('u1:', np.sum(u1))
        print('u2:', np.sum(u2))
        
        print('u0_rep:', np.sum(u0_rep))
        print('u1_rep:', np.sum(u1_rep))
        print('u2_rep:', np.sum(u2_rep))

        print('u0_expl:', np.sum(u0_expl))
        print('u1_expl:', np.sum(u1_expl))
        print('u2_expl:', np.sum(u2_expl))


        fairness = get_Fairness(np.sum(exp0), np.sum(exp1), np.sum(exp2), np.sum(u0), np.sum(u1), np.sum(u2))
        fairness_rep = get_Fairness(np.sum(exp0_rep), np.sum(exp1_rep), np.sum(exp2_rep), np.sum(u0_rep), np.sum(u1_rep), np.sum(u2_rep))
        fairness_expl = get_Fairness(np.sum(exp0_expl), np.sum(exp1_expl), np.sum(exp2_expl), np.sum(u0_expl), np.sum(u1_expl), np.sum(u2_expl))
        
        a_fairness.append(fairness)
        a_fairness_rep.append(fairness_rep)
        a_fairness_expl.append(fairness_expl)

        a_ndcg.append(np.mean(ndcg))
        a_recall.append(np.mean(recall))
        a_hit.append(np.mean(hit))
        a_repeat_ratio.append(np.mean(repeat_ratio))
        a_explore_ratio.append(np.mean(explore_ratio))
        a_recall_repeat.append(np.mean(recall_repeat))
        a_recall_explore.append(np.mean(recall_explore))
        a_hit_repeat.append(np.mean(hit_repeat))
        a_hit_explore.append(np.mean(hit_explore))
        print(ind, np.mean(recall))
        file.write(str(ind)+' '+str(np.mean(recall))+'\n')

    print('basket size:', size)
    print('recall, ndcg, hit:', np.mean(a_recall), np.mean(a_ndcg), np.mean(a_hit))
    print('repeat-explore ratio:', np.mean(a_repeat_ratio), np.mean(a_explore_ratio))
    print('repeat-explore recall', np.mean(a_recall_repeat), np.mean(a_recall_explore))
    print('repeat-explore hit:', np.mean(a_hit_repeat), np.mean(a_hit_explore))
    print('repeat-explore fairness:', np.mean(a_fairness_rep), np.mean(a_fairness_expl))
    print('fairness:', np.mean(a_fairness))

    file.write('basket size: ' + str(size) + '\n')
    file.write('recall, ndcg, hit: '+ str(np.mean(a_recall)) +' ' +str(np.mean(a_ndcg))+' '+ str(np.mean(a_hit)) +'\n')
    file.write('repeat-explore ratio:'+ str(np.mean(a_repeat_ratio)) +' ' +str(np.mean(a_explore_ratio)) +'\n')
    file.write('repeat-explore recall' + str(np.mean(a_recall_repeat)) + ' ' + str(np.mean(a_recall_explore)) +'\n')
    file.write('repeat-explore hit:' + str(np.mean(a_hit_repeat)) + ' ' + str(np.mean(a_hit_explore)) + '\n')
    file.write('repeat-explore fairness:' + str(np.mean(a_fairness_rep)) + ' ' + str(np.mean(a_fairness_expl)) + '\n')
    file.write('fairness:' + str(np.mean(a_fairness)) + '\n')
    return np.mean(a_recall)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_folder', type=str, required=True, help='x')
    parser.add_argument('--fold_list', type=list, required=True, help='x')
    
    args = parser.parse_args()
    pred_folder = args.pred_folder
    fold_list = args.fold_list
    
    eval_file = 'eval_results.txt' #change here
    f = open(eval_file, 'w')
    for dataset in ['instacart', 'tafeng', 'dunnhumby']:
        f.write('############'+dataset+'########### \n')
        get_repeat_eval(pred_folder, dataset, 10, fold_list, f)
        get_repeat_eval(pred_folder, dataset, 20, fold_list, f)
