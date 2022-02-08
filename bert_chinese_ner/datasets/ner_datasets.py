# Download the vocab.text of "bert-base-chinese" at
# "https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt"
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union
import importlib ##

import torch
from torch.utils.data import Dataset

from bert_chinese_ner import ner_config
from .. import ner_config

from bert_chinese_ner.processors.ner_processor import CNerProcessor, convert_examples_to_features
from ..processors.ner_processor import CNerProcessor, convert_examples_to_features
# from bert_chinese_ner.processors.utils_ner import InputFeatures
from ..processors.utils_ner import InputFeatures


importlib.reload(bert_chinese_ner) ##
importlib.reload(ner_config) ##



class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class NerDataset(Dataset):
    def __init__(
        self,
        file_name: Dict,
        processor = CNerProcessor(),
        tokenizer = ner_config.TOKENIZER,
        data_dir: str = ner_config.DATA_DIR,
        mode: Union[str, Split] = Split.train,
        max_seq_length: int = ner_config.MAX_LEN
    ):

        self.file_name = file_name
        self.processor = processor
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.mode = mode
        self.max_seq_length = max_seq_length

        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")

        label_list = self.processor.get_labels()

        if self.mode == Split.train:
            examples = self.processor.get_train_examples(self.data_dir, self.file_name)
        elif self.mode == Split.dev:
            examples = self.processor.get_dev_examples(self.data_dir, self.file_name)
        else:
            examples = self.processor.get_test_examples(self.data_dir, self.file_name)

        self.features = convert_examples_to_features(examples,
                                                     tokenizer=self.tokenizer,
                                                     label_list=label_list,
                                                     max_seq_length=self.max_seq_length,
                                                     pad_on_right=True)


    def __len__(self):
        return len(self.features)


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Convert to Tensors and build dataset
        feature = self.features[i]

        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(feature.attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(feature.token_type_ids, dtype=torch.long)
        label_ids = torch.tensor(feature.label_ids, dtype=torch.long)

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label_ids': label_ids
        }

        return inputs


data_dir = ner_config.DATA_DIR
file_name = FILE_NAME

# train
examples = processor.get_train_examples(data_dir, file_name)
len(examples)
examples[0]
label_list = processor.get_labels()

features = convert_examples_to_features(examples,
                                        tokenizer=ner_config.TOKENIZER,
                                        label_list=label_list,
                                        max_seq_length=ner_config.MAX_LEN,
                                        pad_on_right=True)

len(features) # 20864
features[0].input_ids
features[-1].input_ids

# dev
examples = processor.get_dev_examples(data_dir, file_name)
len(examples)



