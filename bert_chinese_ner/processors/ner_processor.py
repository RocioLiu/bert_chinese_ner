import os
import importlib ##
from typing import List, Optional

import bert_chinese_ner
from .utils_ner import DataProcessor, InputExample, InputFeatures
# from bert_chinese_ner.processors.utils_ner import DataProcessor, InputExample, InputFeatures


# from bert_chinese_ner import ner_config
from .. import ner_config

# importlib.reload(bert_chinese_ner) ##
# importlib.reload(ner_config) ##



class CNerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir, file_name):
        """Returns the training examples from the data directory."""
        return self._create_examples(self._read_text(os.path.join(data_dir, file_name["train"])), "train")

    def get_dev_examples(self, data_dir, file_name):
        """Gets a collection of [`InputExample`] for the dev set."""
        return self._create_examples(self._read_text(os.path.join(data_dir, file_name["dev"])), "dev")

    def get_test_examples(self, data_dir, file_name):
        """Gets a collection of [`InputExample`] for the test set."""
        return self._create_examples(self._read_text(os.path.join(data_dir, file_name["test"])), "test")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ['B-LOC', 'B-ORG', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER', 'O']

    def _create_examples(self, rows, set_type):
        """
        Creates examples for the training and dev sets.

        Args:
            rows: List[Dict['words': List[str], 'labels': List[str]]]
            set_type: str.
        """

        examples = []
        for (i, row) in enumerate(rows):
            guid = f"{set_type}-{i}"
            text = row['words']
            labels = row['labels']
            examples.append(InputExample(guid=guid, text=text, label=labels))

        return examples


def convert_examples_to_features(
        examples: List[InputExample],
        tokenizer=None,
        label_list=None,
        max_seq_length: Optional[int] = None,
        pad_on_right=True):
    """ Loads a data file into a list of `InputBatch`s

        Args:
            pad_on_right: Default to True. Padding the sequence on the right hand side.
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for idx, example in enumerate(examples):
        tokens = example.text
        label_ids = [label_map[l] for l in example.label]
        # Account for [CLS] and [SEP] with "- 2".
        tokens = tokens[:max_seq_length - 2]
        label_ids = label_ids[:max_seq_length - 2]

        input_ids = tokenizer.encode(tokens, add_special_tokens=False)

        input_ids = [101] + input_ids + [102]  # [CLS] + input_ids + [SEP]
        label_ids = [0] + label_ids + [0]

        # attention mask
        # only the real token with value 1 are attends to, and pads tokens with 0.
        mask = [1] * len(input_ids)

        # Segment the two sequences
        token_type_ids = [0] * len(input_ids)

        ## Pad the inputs on the right hand side
        padding_len = max_seq_length - len(input_ids)

        if pad_on_right:
            input_ids = input_ids + [0] * padding_len
            mask = mask + [0] * padding_len
            token_type_ids = token_type_ids + [0] * padding_len
            label_ids = label_ids + [0] * padding_len
        else:
            input_ids = [0] * padding_len + input_ids
            mask = [0] * padding_len + mask
            token_type_ids = [0] * padding_len + token_type_ids
            label_ids = [0] * padding_len + label_ids

        assert len(input_ids) == max_seq_length
        assert len(mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        # {
        #     "text_ids": torch.tensor(text_ids, dtype=torch.long),
        #     "mask": torch.tensor(mask, dtype=torch.long),
        #     "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        #     "tag_ids": torch.tensor(tag_ids, dtype=torch.long)
        # }

        features.append(InputFeatures(input_ids=input_ids,
                                      attention_mask=mask,
                                      token_type_ids=token_type_ids,
                                      label_ids=label_ids))

        return features



# processor = CNerProcessor()
# examples = processor.get_dev_examples(ner_config.DATA_DIR)