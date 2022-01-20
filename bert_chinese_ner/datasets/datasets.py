# Download the vocab.text of "bert-base-chinese" at
# "https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt"
import torch
from os.path import join

from bert_chinese_ner import config
from bert_chinese_ner.datasets import data_processor


importlib.reload(bert_chinese_ner)
importlib.reload(config)



class EntityDataset:
    def __init__(self, data, tokenizer, tag_to_int):
        # data:
        # texts: List[List[str]], [['海', '釣',... ], ['這',...], ...]
        # tags: List[List[str]], [['O', 'O', ...], ['O',...], ...]
        self.texts = data[0]
        self.tags = data[1]
        self.tokenizer = tokenizer
        self.tag_to_int = tag_to_int

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item] # a sample of texts, e.g. ['海', '釣',... ]
        tag = self.tags[item] # ['O', 'O', ...]

        text_ids = config.TOKENIZER.encode(text, add_special_tokens=False)
        tag_ids = [tag_to_int[t] for t in tag]

        # Account for [CLS] and [SEP] with "- 2".
        text_ids = text_ids[:config.MAX_LEN - 2] # we'll add special token [CLS] and [SEP] later
        tag_ids = tag_ids[:config.MAX_LEN - 2]

        text_ids = [101] + text_ids + [102] # [CLS] + text + [SEP]
        tag_ids = [0] + tag_ids + [0]

        # attention mask
        # only the real token with value 1 are attends to, and 0 for padded tokens.
        mask = [1] * len(text_ids)

        # Segment the two sequences
        token_type_ids = [0] * len(text_ids)

        ## Pad the inputs on the right hand side
        padding_len = config.MAX_LEN - len(text_ids)

        text_ids = text_ids + [0] * padding_len
        mask = mask + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len
        tag_ids = tag_ids + [0] * padding_len

        assert len(text_ids) == config.MAX_LEN
        assert len(mask) == config.MAX_LEN
        assert len(token_type_ids) == config.MAX_LEN
        assert len(tag_ids) == config.MAX_LEN

        return {
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "tag_ids": torch.tensor(tag_ids, dtype=torch.long)
        }



#config.TOKENIZER.encode("今天天氣真好！星期二，猴子肚子餓。", add_special_tokens=False)



