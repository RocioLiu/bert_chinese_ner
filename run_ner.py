import torch
from torch.utils.data import DataLoader

from bert_chinese_ner.datasets.ner_datasets import NerDataset
from .datasets.ner_datasets import NerDataset

from bert_chinese_ner import ner_config
from .bert_chinese_ner import ner_config

import importlib
import bert_chinese_ner
importlib.reload(bert_chinese_ner)


MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertCrfForNer, CNerTokenizer),
    'albert': (AlbertConfig, AlbertSoftmaxForNer, CNerTokenizer),
}


tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None,)


FILE_NAME = {"train":ner_config.TRAINING_FILE, "dev":ner_config.DEV_FILE, "test":ner_config.TEST_FILE}



train_dataset = NerDataset(file_name=FILE_NAME, mode="train")
train_dataset[0]['input_ids']

dev_dataset = NerDataset(file_name=FILE_NAME, mode="dev")
len(dev_dataset)

test_dataset = NerDataset(file_name=FILE_NAME, mode="test")


train_dataloader = DataLoader(train_dataset, batch_size=ner_config.TRAIN_BATCH_SIZE,
                              shuffle=True, drop_last=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=ner_config.DEV_BATCH_SIZE,
                              shuffle=True, drop_last=True)