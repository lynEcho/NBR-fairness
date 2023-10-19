import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

data_path = f'../mergedata/{args.dataset}_merged.json'
key_path = f'../keyset/{args.dataset}_keyset.json'
with open(data_path, 'r') as json_file:
    data = json.load(json_file)
with open(key_path, 'r') as json_file:
    key = json.load(json_file)

repeat_ratio = dict()
keyset_user = dict()
test02 = []
test24 = []
test46 = []
test68 = []
test81 = []

for user in key['test']:
    history = data[user][:-1]
    truth = data[user][-1]
    history_set = set([item for sublist in history for item in sublist])
    repeat = 0
    for i in truth:
        if i in history_set:
            repeat += 1

    repeat_ratio[user] = repeat/len(truth)


for key, value in repeat_ratio.items():
    if value <= 0.2:
        test02.append(str(key))
    elif 0.2 < value <= 0.4:
        test24.append(str(key))
    elif 0.4 < value <= 0.6:
        test46.append(str(key))
    elif 0.6 < value <= 0.8:
        test68.append(str(key))
    else:
        test81.append(str(key))


keyset_user['test02'] = test02
keyset_user['test24'] = test24
keyset_user['test46'] = test46
keyset_user['test68'] = test68
keyset_user['test81'] = test81




keyset_file = f'../keyset/{args.dataset}_keyset_user.json'
with open(keyset_file, 'w') as f:
    json.dump(keyset_user, f)


