import torch
from collections import Counter


class SeqEntityScore(object):
    def __init__(self, id2label, markup='bio'):
        self.id2label = id2label
        self.markup = markup
        self.reset()

    def reset(self):


    def compute(self):