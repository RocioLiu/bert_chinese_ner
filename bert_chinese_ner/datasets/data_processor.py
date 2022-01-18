import os
from collections import Counter
import importlib
import opencc

import bert_chinese_ner
from bert_chinese_ner import config

importlib.reload(bert_chinese_ner)
importlib.reload(config)


path = config.ROOT_DIR
train_file = config.TRAINING_FILE
dev_file = config.DEV_FILE


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
    word_counts = Counter(char.lower() for sample in train_data[0] + dev_data[0] for char in sample)

    vocab = [w for w, c in word_counts.items() if c >= min_freq]
    vocab_to_int = {w: (i + 1) for i, w in enumerate(vocab)}
    vocab = ['<PAD>'] + vocab + ['<UNK>']
    vocab_to_int['<UNK>'] = len(vocab_to_int)
    vocab_to_int['<PAD>'] = 0

    ## Build a dictionary that maps tags to integers
    tag_set = set([tag for sample in train[1] + dev[1] for tag in sample])
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









word2idx = dict((w, i) for i, w in enumerate(character))

## Build a dictionary that maps words to integers
words_count = Counter(data["Word"])
vocab_to_int = {w: (i+1) for i, w in enumerate(words_count)}
vocab_to_int['UNK'] = len(vocab_to_int)
vocab_to_int['<PAD>'] = 0







class



class EntityDataset:
        def __init__(self, texts, pos, tags):
            # texts: list of list [["Hi", "," , "my", "name", "is", "rocio"], ["Hello",...."], ...]
            # pos/tags: list of list [[1 2 3 4 5], [...], ...]
            self.texts = texts
            self.pos = pos
            self.tags = tags

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, item):
            text = self.texts[item]  # e.g. ["Hi", "," , "my", "name", "is", "rocio"]
            pos = self.pos[item]
            tags = self.tags[item]

            # The text is already tokenized, but it's not tokenized for BERT,
            # we have to tokenize it for BERT
            ids = []
            target_pos = []
            target_tag = []

            for i, s in enumerate(text):
                inputs = config.TOKENIZER.encode(
                    s,  # one word e.g. rocio
                    add_special_tokens=False  # whether to add the special token as <CLS>, <SEP>
                )
                # word piece e.g. rocio -> ro ##cio
                inputs_len = len(inputs)
                ids.extend(inputs)
                target_pos.extend([pos[i]] * inputs_len)
                target_tag.extend([tags[i]] * inputs_len)

            ids = ids[:config.MAX_LEN - 2]  # because we'll add special token <CLS> and <SEP> later
            target_pos = target_pos[:config.MAX_LEN - 2]
            target_tag = target_tag[:config.MAX_LEN - 2]

            ids = [101] + ids + [102]  # <CLS> + ids + <SEP>
            target_pos = [0] + target_pos + [0]
            target_tag = [0] + target_tag + [0]

            # attention masks
            mask = [1] * len(ids)
            token_type_ids = [0] * len(ids)  # Create a mask from the two sequences

            ## Pad the inputs on the right hand side
            padding_len = config.MAX_LEN - len(ids)

            ids = ids + ([0] * padding_len)
            mask = mask + ([0] * padding_len)
            token_type_ids = token_type_ids + ([0] * padding_len)
            target_pos = target_pos + ([0] * padding_len)
            target_tag = target_tag + ([0] * padding_len)

            return {
                "ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "target_pos": torch.tensor(target_pos, dtype=torch.long),
                "target_tag": torch.tensor(target_tag, dtype=torch.long)
            }

