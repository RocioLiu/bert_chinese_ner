# Download the vocab.text of "bert-base-chinese" at
# "https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt"
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

from bert_chinese_ner import config
from bert_chinese_ner.processors.ner_processor import CNerProcessor, convert_examples_to_features


importlib.reload(bert_chinese_ner)
importlib.reload(config)


@dataclass
class TrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on the command
    line.
    """

    processor = CNerProcessor()
    tokenizer = config.TOKENIZER
    data_dir: str = field(default=config.DATA_DIR)
    max_seq_length: int = field(default=128)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class NerDataset(Dataset):

    args: TrainingArguments
    features: List[InputFeatures]

    def __init__(
        self,
        args: TrainingArguments,
        mode: Union[str, Split] = Split.train
    ):
        self.args = args
        self.mode = mode

        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")

        label_list = args.processor.get_labels()

        if mode == Split.train:
            examples = self.processor.get_train_examples(args.data_dir)
        elif mode == Split.dev:
            examples = self.processor.get_dev_examples(args.data_dir)
        else:
            examples = self.processor.get_test_examples(args.data_dir)

        self.features = convert_examples_to_features(
            examples,
            tokenizer=args.tokenizer,
            label_list=label_list,
            max_seq_length=args.max_seq_length,
            pad_on_right=True)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item) -> InputFeatures:
        return self.features[item]

    def get_labels(self):
        return self.label_list





