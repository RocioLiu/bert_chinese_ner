import os
import json

import bert_chinese_ner
from bert_chinese_ner import config

print(config)

import importlib
importlib.reload(bert_chinese_ner)

os.getcwd()


'Peoples_Daily'

def _parse_data(file_path):
    x_data = []
    y_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        x, y = [], []


path = config.ROOT_DIR
train_file = config.TRAINING_FILE


def load_data(path=None, train_file=None, valid_file=None):
    train_path = os.path.join(path, train_file)
    valid_path = os.path.join(path, valid_file)

