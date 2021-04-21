from numpy.core.fromnumeric import _partition_dispatcher
import pandas as pd
import os
import re
from os.path import join
import codecs
import nltk
from nltk.tokenize import RegexpTokenizer
from config import INDEX_PATH, DATA_PATH, RANDOM_STATE, K


def path2raw(path):
    l = path.split('/')
    path = '/'.join((l[0], 'trec06p', l[1], l[2], l[3]))
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as f:
        # tokenizer = RegexpTokenizer(r'\w+')
        # words = tokenizer.tokenize(f.read())
        # return words
        return f.read()

def raw2content(raw):
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(raw)
    return words

def count_caps(raw):
    return len(re.findall(r'[A-Z]', raw))

def count_letters(raw):
    return len(re.findall(r'[a-zA-Z]', raw))
def gather_and_tokenize():
    print('Gathering raw data...')
    df = pd.read_csv(INDEX_PATH, sep=' ', names=['type', 'path'])
    df['raw'] = df['path'].apply(lambda x: path2raw(x))
    df['content'] = df['raw'].apply(lambda x: raw2content(x))
    df['cap_num'] = df['raw'].apply(lambda x: count_caps(x))
    df['letter_num'] = df['raw'].apply(lambda x: count_letters(x))
    df['cap_ratio'] = df['cap_num'] / df['letter_num']
    df.drop(columns='path', axis=1, inplace=True)
    df.drop(columns='raw', axis=1, inplace=True)
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
    df.to_csv(join(DATA_PATH, 'raw_data.csv'), sep='\t', index=0)

def shuffle_data():
    print('Shuffling data...')
    df = pd.read_csv(join(DATA_PATH, 'raw_data.csv'), sep='\t')
    shuffled_df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index()
    shuffled_df.drop(columns='index', axis=1, inplace=True)
    shuffled_df.to_csv(join(DATA_PATH, 'shuffled_data.csv'), sep='\t', index=0)

def k_fold(k):
    print('Spliting data...')
    df = pd.read_csv(join(DATA_PATH, 'shuffled_data.csv'), sep='\t')
    for i in range(k):
        group_df = df[df.index % k == i]
        group_df.to_csv(join(DATA_PATH, 'group{}.csv').format(i), sep='\t', index=0)


gather_and_tokenize()
shuffle_data()
k_fold(K)