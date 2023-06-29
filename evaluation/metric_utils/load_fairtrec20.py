import pandas as pd
import glob


# loading participants' run
def load_recs(path):
    path = path
    rec_df = []
    for file in glob.iglob(path + '*.parquet'):
        if 'author' in file or 'document' in file:
            continue
        rec_df.append(pd.read_parquet(file))
    return pd.concat(rec_df, ignore_index=True)


# loading document relenace with author information
def load_test(authors, path):
    ratings = load_ratings(path)
    #authors = load_authors(path)
    test_rates = ratings.merge(authors, on='docID', how='inner')
    test_rates.rename(columns={'participant': 'system', 'docID': 'item', 'relevance': 'rating'}, inplace=True)
    if 'rating' not in test_rates.columns:
        test_rates['rating'] = 1
    # test_rates = setup_group(test_rates)
    return test_rates


# loading author info
def load_authors(path):
    authors = pd.read_parquet(path)
    #authors.drop(columns={'Unnamed: 0'}, inplace=True)
    return authors


# loading relevance info
def load_ratings(path):
    #path = 'data/trec2020-fair-archive/submission_parquet/document_relevance.parquet'
    ratings = pd.read_parquet(path)
    ratings.rename(columns={'doc_id': 'docID'}, inplace=True)
    return ratings


# loading sensitive attribute
# def setup_group(rec_df):
#     rec_df['econ_dev'] = 'Unknown'
#     rec_df.loc[(rec_df['Developing'] == 1.0), 'econ_dev'] = 'Developing'
#     rec_df.loc[(rec_df['Advanced'] == 1.0), 'econ_dev'] = 'Advanced'
#     return rec_df


# loading participants' run with author and relevance info
def process_recs(rec_df, authors, path):
    #authors = load_authors(path)
    ratings = load_ratings(path)
    ratings_set = rec_df.merge(ratings, on=['qid', 'docID'], how='inner')
    rec_df = ratings_set.merge(authors, on='docID', how='inner')
    rec_df.rename(columns={'participant': 'system', 'docID': 'item',
                           'relevance': 'rating', 'q_num':'sequence'}, inplace=True)
    rec_df['rank'] = rec_df['rank'].astype(int)
    #rec_df['user'] = rec_df['q_num']
    # setup_group(rec_df)
    return rec_df


def G(authors):
    authors.rename(columns={'docID': 'item'}, inplace=True)
    #authors = load_authors(path)
    # authors = setup_group(authors)
    return pd.Series(authors[['Advanced', 'Developing', 'Unknown']].sum())
