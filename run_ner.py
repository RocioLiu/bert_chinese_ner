import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader


from bert_chinese_ner import ner_config
from bert_chinese_ner.datasets.ner_datasets import NerDataset
from bert_chinese_ner.processors.ner_processor import CNerProcessor
from bert_chinese_ner.models.bert_for_ner import BertCrfForNer
from bert_chinese_ner.callbacks.optimizer import AdamW
from bert_chinese_ner.callbacks.lr_scheduler import get_linear_schedule_with_warmup
from bert_chinese_ner.metrics.ner_metrics import f1_score_func
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
            # transpose the shape to (seq_len, batch_size) xxx
            data[k] = v.to(device)
        optimizer.zero_grad()
        logits, loss = model(**data)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), ner_config.GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def eval_fn(data_loader, model, device):

    model.eval()
    total_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)

        with torch.no_grad():
            logits, loss = model(**data)
            # pred_tags: (nbest, batch_size, seq_length)
            pred_tags = model.crf.decode(logits, data['attention_mask'])

        total_loss += loss.item()

        f1_score = f1_score_func(y_true=data['label_ids'],
                                 y_pred=pred_tags,
                                 mask=data['attention'])

    return total_loss / len(data_loader)



def predict_fn(data_loader, model):
    model.eval()
    total_loss = 0

    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)

        with torch.no_grad():
            loss, outputs = model(**data)

        total_loss += loss.item() # has loss or not

    return



FILE_NAME = {"train":ner_config.TRAINING_FILE, "dev":ner_config.DEV_FILE, "test":ner_config.TEST_FILE}
label_list = CNerProcessor().get_labels()
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for i, label in enumerate(label_list)}

train_dataset = NerDataset(file_name=FILE_NAME, mode="train")
dev_dataset = NerDataset(file_name=FILE_NAME, mode="dev")
test_dataset = NerDataset(file_name=FILE_NAME, mode="test")


train_dataloader = DataLoader(train_dataset, batch_size=ner_config.TRAIN_BATCH_SIZE,
                              shuffle=True, drop_last=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=ner_config.DEV_BATCH_SIZE,
                            shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=ner_config.TEST_BATCH_SIZE,
                            shuffle=True, drop_last=True)



