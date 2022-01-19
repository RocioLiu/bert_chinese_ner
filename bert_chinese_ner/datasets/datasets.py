import torch

from bert_chinese_ner import config
from bert_chinese_ner.datasets import data_processor


class EntityDataset:
    def __init__(self, data, tag_to_int):
        # data: texts: List[List[char]], tags: List[List[tag]]
        self.texts = data[0]
        self.tags = data[1]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        tag = self.tags[item]





class EntityDataset2:
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