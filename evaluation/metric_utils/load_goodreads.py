import pandas as pd
import os
import glob
import sys

def load_test(dname, set='GR-I'):
    test = pd.read_parquet(dname+'/'+set+'/'+'test-ratings.parquet')
    test.set_index(['user', 'item'], drop=True, inplace=True)
    if 'rating' not in test.columns:
        test['rating'] = 1.0
    return test

def gender_stats(data):
    data['male'] = (data['gender'] == 'male').astype('int32')
    data['female'] = (data['gender'] == 'female').astype('int32')
    data['unknown'] = (data['gender'] == 'unknown').astype('int32')
    return data

def join_recs(recs, dname):
    book_gender = load_authors()
    recs = recs.join(book_gender, on='item', how='left')
    recs['gender'].fillna('unknown', inplace=True)
    recs = recs.replace('ambiguous', 'unknown')
    test = load_test(dname)
    recs = recs.join(test['rating'], on=['user', 'item'], how='left')
    return recs

def load_recs(dname, set='GR-I'):
    dir = dname+'/'+set
    recs = []
    for file in glob.glob(os.path.join(dir, '*-recs.parquet')):
        if 'reranking' in file:
            continue
        algo = file[len(dir)+1:-len('-recs.parquet')]
        recs.append(pd.read_parquet(file).assign(Algorithm=algo, Set=set))
    return pd.concat(recs, ignore_index=True)

def load_authors():
    #book_gender = pd.read_parquet('data/cluster-genders.csv.gz')
    book_gender = pd.read_parquet('data/cluster-genders.parquet')
    book_gender = book_gender.set_index('cluster')['gender']
    book_gender.index.name = 'item'
    book_gender.loc[book_gender.str.startswith('no-')] = 'unknown'
    book_gender.loc[book_gender == 'unlinked'] = 'unknown'
    book_gender = book_gender.astype('category')
    return book_gender

def name_change(x):
    if x.startswith('no-'):
        x = 'unknown'
    if x =='ambiguous':
        x='unknown'
    else:
        return x
    return x



def load_relevance(dname, set='GR-I'):
    dir = dname+'/'+set
    relev = []
    print(os.path.join(dir, '*-group-relevance.csv.gz'))

    print(glob.glob('eval5/GR-I/*-group-relevance.csv.gz'))

    print(glob.glob(os.path.join(dir, '*-group-relevance.csv.gz')))
    sys.exit()
    for file in glob.glob(os.path.join(dir, '*-group-relevance.csv.gz')):
        print(file)
        if 'test-rating' in file:
            continue
        algo = file[len(dir)+1:-len('-group-relevance.csv.gz')]
        print(algo)
        relev.append(pd.read_csv(file).assign(Algorithm=algo, Set=set))
    print(relev)
    relev_df = pd.concat(relev, ignore_index=True)
    relev_df.drop(columns=['Unnamed: 0'],inplace=True)
    relev_df = relev_df.melt(id_vars=['user', 'Algorithm', 'Set'],var_name='gender', value_name='score')
    relev_df['gender'] = relev_df['gender'].apply(lambda x: name_change(x))
    return relev_df

# use this to call G in notebooks
def G():
    counts = pd.read_csv('data/gender-statistics.csv')
    gri = counts[counts['DataSet'] == 'GR-I']
    female = gri['female']
    male = gri['male']
    #unknown = gri['unknown'] + gri['ambiguous'] + gri['no-viaf-author'] + gri['no-loc-author']
    unknown = gri['unknown'] + gri['ambiguous'] + gri['no-book-author'] + gri['no-author-rec']+gri['no-book']

    return pd.Series({'female': female, 'male': male, 'unknown': unknown})

# use this for processing test sets in notebook
def process_tests(dname):
    book_gender = load_authors()
    test_set = load_test(dname)
    test_rates = test_set.reset_index().join(book_gender, on='item', how='left')
    test_rates['gender'].fillna('unknown', inplace=True)
    test_rates = test_rates.replace('ambiguous', 'unknown')
    test_rates = gender_stats(test_rates)
    return test_rates

# use this for rec data in notebook
def process_recs(dname):
    recs = load_recs(dname)
    recs = join_recs(recs, dname)
    recs = gender_stats(recs)
    recs['rating'].fillna(0, inplace=True)
    return recs
