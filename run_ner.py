import torch
from torch.utils.data import DataLoader

# from bert_chinese_ner.datasets.ner_datasets import NerDataset
from .datasets.ner_datasets import NerDataset

# from bert_chinese_ner import ner_config
from .bert_chinese_ner import ner_config


MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertSoftmaxForNer, CNerTokenizer),
    'albert': (AlbertConfig, AlbertSoftmaxForNer, CNerTokenizer),
}


tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None,)


FILE_NAME = {"train":ner_config.TRAINING_FILE, "dev":ner_config.DEV_FILE, "test":ner_config.TEST_FILE}

train_dataset = NerDataset()


train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE,
                              shuffle=True, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE,
                              shuffle=True, drop_last=True)