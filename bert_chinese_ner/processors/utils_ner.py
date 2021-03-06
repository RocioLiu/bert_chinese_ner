import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union

import opencc


@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text: list. The words of the sequence.
        label: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text: List[str]
    label: Optional[List[str]] = None


@dataclass
class InputFeatures:
    """
    A single set of features of data. Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second
            portions of the inputs.
        label_ids: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    label_ids: List[int]


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir, file_name):
        """Gets a collection of [`InputExample`] for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, file_name):
        """Gets a collection of [`InputExample`] for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir, file_name):
        """Gets a collection of [`InputExample`] for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_text(self, input_file):
        """Reads a text file."""
        rows = []
        with open(input_file, 'r', encoding="utf-8") as f:
            # a sequence of text which represents one example.
            # e.g. ['???', '???', ..., '???']
            words = []
            # a sequence of label which represents one example
            # e.g. ['O', 'O', ..., 'O']
            labels = []
            lines = f.read().splitlines()
            # simplified chinese -> traditional chinese
            converter = opencc.OpenCC('s2t')
            lines = converter.convert('\n'.join(lines)).split('\n')
            # lines: ['??? O', '??? O', ..., '??? O', '??? O']
            for line in lines:
                if line == "" or line == "\n":
                    if words:
                        rows.append({"words":words, "labels":labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        # e.g. splits: ['???', 'O']
                        labels.append(splits[-1])
                    else:
                        # Examples could have no label for mode = "test"
                        # e.g. splits: ['???']
                        labels.append("O")

        # data: List[Dict['words': List[str], 'labels': List[str]]]

        return rows



