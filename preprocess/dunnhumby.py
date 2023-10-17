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
note: order number may skip, but guarantes sequential information
'''

input_file_path = '/ivi/ilps/personal/yliu10/NBR-fairness/rawdata/dunnhumby/transaction_data.csv'
#product_path = '/ivi/ilps/personal/yliu10/NBR-fairness/rawdata/dunnhumby/product.csv'

train_baskets_file_path = '/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/dunnhumby/fold0/dunnhumby_train_0.csv'
test_baskets_file_path = '/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/dunnhumby/fold0/dunnhumby_test_0.csv'
valid_baskets_file_path = '/ivi/ilps/personal/yliu10/NBR-fairness/csvdata/dunnhumby/fold0/dunnhumby_valid_0.csv'

'''
category = pd.read_csv(product_path, usecols=['PRODUCT_ID', 'DEPARTMENT']).rename(columns={'PRODUCT_ID':'item_id', 'DEPARTMENT':'category'}).drop_duplicates()
category['item_id'] = category['item_id'].astype(str)
print(category.shape)
print(category.nunique())
'''
df = pd.read_csv(input_file_path)
print(df.shape)
df['date'] = df['DAY'].astype(int)
df['basket_id'] = df['BASKET_ID']
df['item_id'] = df['PRODUCT_ID'].astype(str)
df['user_id'] = df['household_key'].astype(str)
df['time'] = df['TRANS_TIME'].astype(int)

processed_df = df[['date','basket_id','user_id','item_id','time']].drop_duplicates()
print(processed_df.shape)
print(processed_df.nunique())

# => user_id,order_number,item_id,basket_id

processed_df.sort_values(['user_id','date','time'], ascending=[True, True, True], inplace=True)
baskets = pd.DataFrame(columns=['user_id','order_number','item_id','basket_id'])


for user, user_data in processed_df.groupby('user_id'):
    user_data.reset_index(drop=True, inplace=True)
    order_num = []
    start = 1
    for i in range(len(user_data['basket_id'])):
        if i > 0:
            if user_data['basket_id'].values[i] != user_data['basket_id'].values[i-1]:
                start += 1 

        order_num.append(start)

    assert len(user_data['basket_id']) == len(order_num)
    user_data['order_number'] = pd.Series(order_num)
    user_data['user_id'] = pd.Series([user for i in range(len(user_data['basket_id']))])

    baskets = baskets.append(user_data[['user_id','order_number','item_id','basket_id']])

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

with open("/ivi/ilps/personal/yliu10/NBR-fairness/preprocess/truncate_dunn_item_ind.json", "w") as outfile:
    json.dump(item_dict, outfile)
with open("/ivi/ilps/personal/yliu10/NBR-fairness/preprocess/truncate_dunn_user_ind.json", "w") as outfile:
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


