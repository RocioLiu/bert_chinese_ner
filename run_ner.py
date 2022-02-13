import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader


from bert_chinese_ner import ner_config
from bert_chinese_ner.datasets.ner_datasets import NerDataset
from bert_chinese_ner.models.bert_for_ner import BertCrfForNer
from bert_chinese_ner.callbacks.optimizer import AdamW
from bert_chinese_ner.callbacks.lr_scheduler import get_linear_schedule_with_warmup
from bert_chinese_ner.models.transformers.models.bert.configuration_bert import BertConfig

import importlib
import bert_chinese_ner
importlib.reload(bert_chinese_ner)
importlib.reload(ner_config)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertCrfForNer, CNerTokenizer),
    'albert': (AlbertConfig, AlbertSoftmaxForNer, CNerTokenizer),
}


def train_fn(data_loader, model, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    progress_bar = tqdm(data_loader, desc=f"Epoch {}")

    # get a batch of data dict
    for data in data_loader:
        for k, v in data.items():
            data[k] = v.transpose(0, 1).to(device)
        optimizer.zero_grad()
        logits, loss = model(**data)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), ner_config.GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    metric =

    model.eval()
    total_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.transpose(0, 1).to(device)

        with torch.no_grad():
            logits, loss = model(**data)

        total_loss += loss.item()
        pred_tags = model.crf.decode(logits, data['attention_mask'])

    return total_loss / len(data_loader)


def predict_fn(data_loader, model):
    model.eval()
    total_loss = 0

    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.transpose(0, 1).to(device)

        with torch.no_grad():
            loss, outputs = model(**data)

        total_loss += loss.item() # has loss or not



FILE_NAME = {"train":ner_config.TRAINING_FILE, "dev":ner_config.DEV_FILE, "test":ner_config.TEST_FILE}

train_dataset = NerDataset(file_name=FILE_NAME, mode="train")
dev_dataset = NerDataset(file_name=FILE_NAME, mode="dev")
test_dataset = NerDataset(file_name=FILE_NAME, mode="test")


train_dataloader = DataLoader(train_dataset, batch_size=ner_config.TRAIN_BATCH_SIZE,
                              shuffle=True, drop_last=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=ner_config.DEV_BATCH_SIZE,
                            shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=ner_config.TEST_BATCH_SIZE,
                            shuffle=True, drop_last=True)



