import os

from .utils_ner import DataProcessor
from bert_chinese_ner.processors.utils_ner import DataProcessor

from bert_chinese_ner import config

importlib.reload(bert_chinese_ner)
importlib.reload(config)


class CNerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """Returns the training examples from the data directory."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "example.train")), "train")

    def get_dev_examples(self, data_dir):
        """Gets a collection of [`InputExample`] for the dev set."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "example.dev")), "dev")

    def get_test_examples(self, data_dir):
        """Gets a collection of [`InputExample`] for the test set."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "example.test")), "test")

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