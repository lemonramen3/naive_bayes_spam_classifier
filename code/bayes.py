from os.path import join, split
import pandas as pd
from pandas.core.algorithms import value_counts
from config import DATA_PATH, K, ALPHA, SAMPLE_RATE, RANDOM_STATE, USE_CAP, THREHOLD, EPS
from collections import Counter
import itertools
from tqdm import tqdm

def data_split(group_index: int):
    print('Spliting data for fold {}...'.format(group_index))
    test_data = pd.read_csv(join(DATA_PATH, 'group{}.csv'.format(group_index)), sep='\t', converters={'content': eval})
    train_data = pd.concat([pd.read_csv(join(DATA_PATH, 'group{}.csv'.format(i)), sep='\t', converters={'content': eval}) for i in range(K) if i != group_index])
    return train_data.sample(frac=SAMPLE_RATE, random_state=RANDOM_STATE), test_data

def word_count(df: pd.DataFrame):
    print('Building vocabulary...')
    spam_df = df[df['type'] == 'spam']
    ham_df = df[df['type'] == 'ham']
    spam_vocab = Counter(list(itertools.chain(*(spam_df['content'].tolist()))))
    ham_vocab = Counter(list(itertools.chain(*(ham_df['content'].tolist()))))
    return spam_vocab, ham_vocab

def naive_bayes(train_data: pd.DataFrame, test_data: pd.DataFrame):
    P_spam = train_data['type'].value_counts()['spam'] / train_data.shape[0]
    P_ham = 1 - P_spam
    spam_vocab, ham_vocab = word_count(train_data)
    N_spam = sum(spam_vocab.values())
    N_ham = sum(ham_vocab.values())
    N_vocab = len(spam_vocab) + len(ham_vocab)
    print('Number of words in spam: ', N_spam)
    print('Number of words in ham: ', N_ham)
    print('Vocabulary size: ', N_vocab)

    def apply_cap_feature(test_cap_ratio):
        threhold = THREHOLD
        eps = EPS
        spam_df = train_data[train_data['type'] == 'spam']
        ham_df = train_data[train_data['type'] == 'ham']
        P_cap_spam = spam_df[abs(spam_df['cap_ratio'] - test_cap_ratio) < threhold].shape[0] / spam_df.shape[0] + eps
        P_cap_ham = ham_df[abs(ham_df['cap_ratio'] - test_cap_ratio) < threhold].shape[0] / ham_df.shape[0] + eps
        return P_cap_spam / P_cap_ham

    def predict_spam(word_list: list):
        predict_spam = P_spam
        predict_ham = P_ham
        predict = predict_spam / predict_ham
        for word in word_list:
            N_wi_spam = spam_vocab[word]
            N_wi_ham = ham_vocab[word]
            P_wi_spam = (N_wi_spam + ALPHA) / (N_spam + ALPHA * N_vocab)
            P_wi_ham = (N_wi_ham + ALPHA) / (N_ham + ALPHA * N_vocab)
            predict *= P_wi_spam / P_wi_ham
        return predict
    
    def translate(predict):
        return 'spam' if predict > 0.5 else 'ham'
    test_data['predict'] = test_data['content'].apply(lambda x: predict_spam(x))
    if USE_CAP:
        test_data['predict_cap_ratio'] = test_data['cap_ratio'].apply(lambda x: apply_cap_feature(x))
        test_data['predict'] = test_data['predict'] * test_data['predict_cap_ratio'] * test_data['letter_num']


    test_data['predict'] = test_data['predict'].apply(lambda x: translate(x))
    true_pos = test_data[(test_data['type'] == 'ham') & (test_data['predict'] == 'ham')].shape[0]
    true_neg = test_data[(test_data['type'] == 'spam') & (test_data['predict'] == 'spam')].shape[0]
    false_pos = test_data[(test_data['type'] == 'spam') & (test_data['predict'] == 'ham')].shape[0]
    false_neg = test_data[(test_data['type'] == 'ham') & (test_data['predict'] == 'spam')].shape[0]
    test_size = test_data.shape[0]
    accuracy = (true_pos + true_neg) / test_size
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = (2 * true_pos) / (2 * true_pos + false_pos + false_neg)
    print('Test data size: ', test_size)
    print('Accuracy      : ', accuracy)
    print('Precision     : ', precision)
    print('Recall        : ', recall)
    print('F1            : ', f1)
    return true_pos, true_neg, false_pos, false_neg, test_size


if __name__ == '__main__':
    print('Training configuration: training data sample rate = {}, alpha = {}, random state = {}, {} folds in total.'.format(SAMPLE_RATE, ALPHA, RANDOM_STATE, K))
    true_pos = true_neg = false_pos = false_neg = test_size = 0
    for k in range(K):
        print('==========Fold {}=========='.format(k))
        train_data, test_data = data_split(k)
        true_pos_, true_neg_, false_pos_, false_neg_, test_size_ = naive_bayes(train_data, test_data)
        true_pos += true_pos_
        true_neg += true_neg_
        false_pos += false_pos_
        false_neg += false_neg_
        test_size += test_size_
    accuracy = (true_pos + true_neg) / test_size
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = (2 * true_pos) / (2 * true_pos + false_pos + false_neg)
    print('==========================')
    print('Total test data size: ', test_size)
    print('Total accuracy      : ', accuracy)
    print('Total precision     : ', precision)
    print('Total recall        : ', recall)
    print('Total F1            : ', f1)
