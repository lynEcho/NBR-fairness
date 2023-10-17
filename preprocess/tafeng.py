import pandas as pd
import numpy as np
import json
import sys

'''
Reads the raw files, renames columns;
for train baskets, remove users with fewer than 3 baskets and items bought fewer than 5 times;
output all baskets for recanet
keep their last 50/100 baskets as training set
last basket as test(50%) and validation(50%), the rest as train.
'''

user_order = pd.read_csv('/ivi/ilps/personal/yliu10/NBR-fairness/rawdata/tafeng/ta_feng_all_months_merged.csv',
                         usecols=['TRANSACTION_DT', 'CUSTOMER_ID', 'PRODUCT_ID'])

train_baskets_file_path = '/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/tafeng/fold0/tafeng_train_0.csv'
test_baskets_file_path = '/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/tafeng/fold0/tafeng_test_0.csv'
valid_baskets_file_path = '/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/tafeng/fold0/tafeng_valid_0.csv'

user_order = user_order.dropna(how='any')
print(user_order.shape)
print(user_order.nunique())
user_order['TRANSACTION_DT'] = pd.to_datetime(user_order['TRANSACTION_DT'].astype(str), format='%m/%d/%Y')
print(user_order)

user_order = user_order.rename(columns={'CUSTOMER_ID':'user_id', 'PRODUCT_ID':'item_id'}).drop_duplicates()
print(user_order.nunique())


#'TRANSACTION_DT' ==> 'order_number'

baskets = pd.DataFrame(columns=['user_id', 'order_number', 'item_id', 'basket_id'])
basket_id = 1
for user, user_data in user_order.groupby('user_id'):
    #print(user_data)
    date_list = list(set(user_data['TRANSACTION_DT'].tolist()))
    date_list = sorted(date_list) #from early to latest
    #print(date_list)

    date_num = 1 #start from 1
    for date in date_list:
        date_data = user_data[user_data['TRANSACTION_DT'].isin([date])]
        #print(date_data)
        date_item = list(set(date_data['item_id'].tolist()))
        #print(date_item)
        item_num = len(date_item)
        date_baskets = pd.DataFrame({'user_id': pd.Series([user for i in range(item_num)]),
                                    'order_number': pd.Series([date_num for i in range(item_num)]),
                                    'item_id': pd.Series(date_item),
                                    'basket_id': pd.Series([basket_id for i in range(item_num)])})

        baskets = baskets.append(date_baskets)   
        date_num += 1
        basket_id += 1

print(baskets)
print(baskets.nunique())

#generate train baskets, test baskets

baskets.sort_values(['user_id', 'order_number'], ascending=[True, False], inplace=True)
print(baskets)

last_baskets = baskets.drop_duplicates(subset='user_id', keep='first')[['user_id', 'order_number']]
print(last_baskets)

test_baskets = pd.merge(last_baskets, baskets, how='left') #last-1
print(test_baskets)

train_baskets = pd.concat([baskets,test_baskets]).drop_duplicates(keep=False) #remove last-1
print(train_baskets)

#for train baskets, remove users with fewer than 3 baskets and items bought fewer than 5 times, many times till satisfying the conditions
user_num = train_baskets['user_id'].nunique()
item_count = train_baskets['item_id'].nunique()

while True:
    user_basket_counts = train_baskets.groupby('user_id')['order_number'].nunique()
    item_purchase_counts = train_baskets.groupby('item_id')['order_number'].count()

    valid_users = user_basket_counts[user_basket_counts >= 3].index
    valid_items = item_purchase_counts[item_purchase_counts >= 5].index

    train_baskets = train_baskets[train_baskets['user_id'].isin(valid_users) & train_baskets['item_id'].isin(valid_items)].reset_index(drop=True) 

    if (user_num == train_baskets['user_id'].nunique()) & (item_count == train_baskets['item_id'].nunique()):
        
        filtered_df = train_baskets
        break
    else:
        user_num = train_baskets['user_id'].nunique()
        item_count = train_baskets['item_id'].nunique()
        


print(filtered_df.nunique())
print(filtered_df.shape)


#for test baskets, remove corresponding users

filtered_test = test_baskets[test_baskets['user_id'].isin(valid_users)].reset_index(drop=True)
print(filtered_test.nunique())
print(filtered_test.shape)

#for train baskets, keep last 50 baskets
    
filtered_df.sort_values(['user_id','order_number'], ascending=[True, True], inplace=True)

truncate_baskets = pd.DataFrame(columns=['user_id', 'order_number', 'item_id', 'basket_id'])

for user, user_data in filtered_df.groupby('user_id'):

    #print(user_data)
    order_list = sorted(list(set(user_data['order_number'])))
    order_num = len(order_list)
    if order_num > 50:
        user_data = user_data[user_data['order_number'].isin(order_list[-50:])]
        assert len(set(user_data['order_number'])) == 50

    truncate_baskets = truncate_baskets.append(user_data[['user_id','order_number','item_id','basket_id']])
    
truncate_baskets.sort_values(['user_id','order_number'], ascending=[True, True], inplace=True)
truncate_baskets.reset_index(drop=True,inplace=True) 

print(truncate_baskets.nunique())
print(truncate_baskets.shape)
basket_num = truncate_baskets.groupby('user_id')['order_number'].nunique().sum()
print("basket_num:", basket_num)

#reset the user_id and product_id, replace!

item_dict = dict()
item_ind = 1
user_dict = dict()
user_ind = 1
for ind in range(len(truncate_baskets)):
    item_id = truncate_baskets.at[ind, 'item_id']
    if item_id not in item_dict.keys():
        item_dict[item_id] = item_ind
        item_ind += 1
    truncate_baskets.at[ind, 'item_id'] = item_dict[item_id]

    user_id = truncate_baskets.at[ind, 'user_id']
    if user_id not in user_dict.keys():
        user_dict[user_id] = user_ind
        user_ind += 1
    truncate_baskets.at[ind, 'user_id'] = user_dict[user_id]

for ind in range(len(filtered_test)):
    item_id = filtered_test.at[ind, 'item_id']
    if item_id not in item_dict.keys():
        item_dict[item_id] = item_ind
        item_ind += 1
    filtered_test.at[ind, 'item_id'] = item_dict[item_id]

    user_id = filtered_test.at[ind, 'user_id']
    if user_id not in user_dict.keys():
        user_dict[user_id] = user_ind
        user_ind += 1
    filtered_test.at[ind, 'user_id'] = user_dict[user_id]


with open("/ivi/ilps/personal/yliu10/NBR-fairness/preprocess/truncate_ta_item_ind.json", "w") as outfile:
    json.dump(item_dict, outfile)
with open("/ivi/ilps/personal/yliu10/NBR-fairness/preprocess/truncate_ta_user_ind.json", "w") as outfile:
    json.dump(user_dict, outfile)

#last basket as test(50%) and validation(50%), the rest as train

all_users = list(set(filtered_test['user_id'].tolist())) 
print(len(all_users))

valid_indices = np.random.choice(range(len(all_users)),int(0.5*len(all_users)),
                                 replace=False)
print(len(valid_indices))

valid_users = [all_users[i] for i in valid_indices]

valid_baskets = filtered_test[filtered_test['user_id'].isin(valid_users)]
test_baskets = filtered_test[~filtered_test['user_id'].isin(valid_users)]

print(valid_baskets.shape) 
print(test_baskets.shape) 
print(truncate_baskets.shape) 

print(valid_baskets.nunique())
print(test_baskets.nunique())
print(truncate_baskets.nunique())

truncate_baskets.to_csv(train_baskets_file_path,index=False)
test_baskets.to_csv(test_baskets_file_path,index=False)
valid_baskets.to_csv(valid_baskets_file_path,index=False)