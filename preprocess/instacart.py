import pandas as pd
import numpy as np
import json
import sys
import random

'''
Reads the raw files, renames columns;
sample 20,000 users before filtering 
for train baskets, remove users with fewer than 3 baskets and items bought fewer than 5 times;
output all baskets for recanet
keep their last 50/100 baskets as training set
last basket as test(50%) and validation(50%), the rest as train.
'''


prior_orders_file_path = '/ivi/ilps/personal/yliu10/NBR-fairness/rawdata/instacart/order_products__prior.csv'
train_orders_file_path = '/ivi/ilps/personal/yliu10/NBR-fairness/rawdata/instacart/order_products__train.csv'
orders_file_path = '/ivi/ilps/personal/yliu10/NBR-fairness/rawdata/instacart/orders.csv'
#product_path = '/ivi/ilps/personal/yliu10/NBR-fairness/rawdata/instacart/products.csv'
train_baskets_file_path = '/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/instacart/fold0/instacart_train_0.csv'
test_baskets_file_path = '/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/instacart/fold0/instacart_test_0.csv'
valid_baskets_file_path = '/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/instacart/fold0/instacart_valid_0.csv'

'''
category = pd.read_csv(product_path, usecols=['product_id', 'department_id']).rename(columns={'product_id':'item_id', 'department_id':'category'}).drop_duplicates()
print(category.shape)
print(category.nunique())
'''
prior_orders = pd.read_csv(prior_orders_file_path)
train_orders = pd.read_csv(train_orders_file_path)
all_orders = pd.concat([prior_orders,train_orders])
print(all_orders.shape, flush=True)
print(all_orders.nunique(), flush=True)

order_info = pd.read_csv(orders_file_path)

all_orders = pd.merge(order_info,all_orders,how='inner')
print(all_orders.shape)
print(all_orders.nunique(), flush=True)
print(all_orders.head())

all_orders = all_orders.rename(columns={'order_id':'basket_id', 'product_id':'item_id'}).drop_duplicates()
all_orders = all_orders[['user_id','order_number','item_id','basket_id']]
print(all_orders.shape)
print(all_orders.nunique())
print(all_orders.head())


#sample 20,000 users before filtering
all_user_id = list(set(all_orders['user_id']))
sample_users = random.sample(all_user_id, 20000)

all_orders = all_orders[all_orders['user_id'].isin(sample_users)].reset_index(drop=True)

print(all_orders.shape)
print(all_orders.nunique())
print(all_orders.head())


#generate train baskets, test baskets

all_orders.sort_values(['user_id', 'order_number'], ascending=[True, False], inplace=True)
print(all_orders)

last_baskets = all_orders.drop_duplicates(subset='user_id', keep='first')[['user_id','order_number']]
print(last_baskets)

test_baskets = pd.merge(last_baskets, all_orders, how='left') #last-1
print(test_baskets)
train_baskets = pd.concat([all_orders,test_baskets]).drop_duplicates(keep=False) #remove last-1
print(train_baskets)



#for train baskets, remove users with fewer than 3 baskets and items bought fewer than 5 times

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
        

print(filtered_df.shape)
print(filtered_df.nunique())

#for test baskets, remove corresponding users
filtered_test = test_baskets[test_baskets['user_id'].isin(valid_users)].reset_index(drop=True)
print(filtered_test.nunique())
print(filtered_test.shape)
print(filtered_test)



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

#reset the user_id and product_id

item_dict = dict()
item_ind = 1
user_dict = dict()
user_ind = 1
for ind in range(len(truncate_baskets)):
    item_id = truncate_baskets.at[ind, 'item_id']
    if str(item_id) not in item_dict.keys():
        item_dict[str(item_id)] = item_ind
        item_ind += 1
    truncate_baskets.at[ind, 'item_id'] = item_dict[str(item_id)]

    user_id = truncate_baskets.at[ind, 'user_id']
    if str(user_id) not in user_dict.keys():
        user_dict[str(user_id)] = user_ind
        user_ind += 1
    truncate_baskets.at[ind, 'user_id'] = user_dict[str(user_id)]

for ind in range(len(filtered_test)):
    item_id = filtered_test.at[ind, 'item_id']
    if str(item_id) not in item_dict.keys():
        item_dict[str(item_id)] = item_ind
        item_ind += 1
    filtered_test.at[ind, 'item_id'] = item_dict[str(item_id)]

    user_id = filtered_test.at[ind, 'user_id']
    if str(user_id) not in user_dict.keys():
        user_dict[str(user_id)] = user_ind
        user_ind += 1
    filtered_test.at[ind, 'user_id'] = user_dict[str(user_id)]


with open("/ivi/ilps/personal/yliu10/NBR-fairness/preprocess/sample_ins_item_ind.json", "w") as outfile:
    json.dump(item_dict, outfile)
with open("/ivi/ilps/personal/yliu10/NBR-fairness/preprocess/sample_ins_user_ind.json", "w") as outfile:
    json.dump(user_dict, outfile)


#last basket as test(50%) and validation(50%), the rest as train.

all_users = list(set(filtered_test['user_id'].tolist()))
valid_indices = np.random.choice(range(len(all_users)),int(0.5*len(all_users)),
                                 replace=False)
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
