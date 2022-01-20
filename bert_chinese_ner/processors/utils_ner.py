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


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data. Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in `[0, 1]`: Usually `1` for tokens that are NOT MASKED, `0` for MASKED (padded)
            tokens.
        token_type_ids: Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: List[int] = None
    token_type_ids: List[int] = None
    label: Optional[Union[int, float]] = None


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of [`InputExample`] for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of [`InputExample`] for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of [`InputExample`] for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    @classmethod
    def _read_text(self, input_file):
        """Reads a text file."""
        data = []
        with open(input_file, 'r', encoding="utf-8") as f:
            words = []
            labels = []
            lines = f.read().splitlines()
            # simplified chinese -> traditional chinese
            converter = opencc.OpenCC('s2t.json')
            lines = converter.convert('\n'.join(lines)).split('\n')
            # lines: ['海 O', '釣 O', ..., '域 O', '。 O']
            for line in lines:
                if line == "" or line == "\n":
                    if words:
                        data.append({"words":words, "labels":labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        # e.g. splits: ['海', 'O']
                        labels.append(splits[-1])
                    else:
                        # Examples could have no label for mode = "test"
                        # e.g. splits: ['海']
                        labels.append("O")

        return data



