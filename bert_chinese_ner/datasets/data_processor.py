import os
from collections import Counter
import importlib
import opencc

import bert_chinese_ner
from bert_chinese_ner import config

importlib.reload(bert_chinese_ner)
importlib.reload(config)


converter = opencc.OpenCC('s2t.json')

def _parse_data(file_path, converter,  text_index=0, label_index=1):
    """
    Convert the raw text data to x_data:[list of lists(sentences composed of char)]),
    and y_data:[list of lists(corresponding labels of each sentences)]
    :param file_path:
    :param converter:
    :param text_index:
    :param label_index:
    :return:
    """
    x_data, y_data = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        # simplified chinese -> traditional chinese
        lines = converter.convert('\n'.join(lines)).split('\n')
        # lines: ['在 O', '這 O', ..., '罪 O']
        one_samp_x, one_samp_y = [], []
        for line in lines:
            # line: '在 O'
            # '在 O' -> ['在', 'O'], '這 O' -> ['這', 'O'],...
            row = line.split(' ')
            if len(row) == 1:
                x_data.append(one_samp_x) # Add a complete sample to x_data
                y_data.append(one_samp_y)
                one_samp_x = [] # reset the sample container
                one_samp_y = []
            else:
                one_samp_x.append(row[text_index]) # extend the sample with a char
                one_samp_y.append(row[label_index])
    return x_data, y_data


def _build_vocab(train_data, dev_data, min_freq):
    ## Build a vocab_to_int dict that maps words to integers
    char_counts = Counter(char.lower() for sample in train_data[0] + dev_data[0] for char in sample)

    vocab = [char for char, count in char_counts.items() if count >= min_freq]
    vocab_to_int = {w: (i + 1) for i, w in enumerate(vocab)}
    vocab = ['<PAD>'] + vocab + ['<UNK>']
    vocab_to_int['<UNK>'] = len(vocab_to_int)
    vocab_to_int['<PAD>'] = 0

    ## Build a dictionary that maps tags to integers
    tag_set = set([tag for sample in train_data[1] + dev_data[1] for tag in sample])
    tag_to_int = {t: (i+1) for i, t in enumerate(tag_set)}
    tag_to_int['<PAD>'] = len(tag_to_int)

    return vocab_to_int, tag_to_int



def preprocess_data(path=None, train_file=None, dev_file=None, min_freq=2):
    train_path = os.path.join(path, train_file)
    dev_path = os.path.join(path, dev_file)

    train = _parse_data(train_path, converter)
    dev = _parse_data(dev_path, converter)

    vocab_to_int, tag_to_int = _build_vocab(train, dev, min_freq)

    return train, dev, vocab_to_int, tag_to_int





