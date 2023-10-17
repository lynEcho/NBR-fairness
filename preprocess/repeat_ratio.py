import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

data_path = f'/ivi/ilps/personal/yliu10/NBR-fairness/mergedata/{args.dataset}_merged.json'
key_path = f'/ivi/ilps/personal/yliu10/NBR-fairness/keyset/{args.dataset}_keyset_0.json'
with open(data_path, 'r') as json_file:
    data = json.load(json_file)
with open(key_path, 'r') as json_file:
    key = json.load(json_file)

repeat_ratio = []

for user in key['test']:
    history = data[user][:-1]
    truth = data[user][-1]
    history_set = set([item for sublist in history for item in sublist])
    repeat = 0
    for i in truth:
        if i in history_set:
            repeat += 1

    repeat_ratio.append(repeat/len(truth))

print("Average:", sum(repeat_ratio)/len(repeat_ratio))




