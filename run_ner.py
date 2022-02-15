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
    'bert': (BertConfig, BertCrfForNer, ner_config.TOKENIZER),
    # 'albert': (AlbertConfig, AlbertSoftmaxForNer, ner_config.TOKENIZER),
}

FILE_NAME = {"train":ner_config.TRAINING_FILE, "dev":ner_config.DEV_FILE, "test":ner_config.TEST_FILE}

# ----

processor = CNerProcessor()

label_list = CNerProcessor().get_labels()
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for i, label in enumerate(label_list)}


best_f1 = 0
steps = 0
for epoch in tqdm(range(ner_config.EPOCHS)):

    train_loss, steps = training_fn(train_dataloader, dev_dataloader, model, optimizer, scheduler, epoch, device, steps)

    if eval_f1 > best_f1:
        torch.save(model.state_dict(), ner_config.MODEL_PATH)
        best_f1 = eval_f1



def training_fn(train_dataloader, dev_dataloader,
                model, optimizer, scheduler,
                epoch, device, steps):

    model.train()
    total_train_loss = 0

    progress_bar = tqdm(data_loader, desc=f"Epoch {}")

    # get a batch of data dict
    for data in train_dataloader:

        steps += 1
        for k, v in data.items():
            # transpose the shape to (seq_len, batch_size) xxx
            data[k] = v.to(device)
        optimizer.zero_grad()
        train_logits, train_loss = model(**data)
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), ner_config.GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        total_train_loss += train_loss.item()

        if steps % ner_config.PRINT_EVERY_N_STEP == 0:
            model.eval()
            y_true_list = []
            y_pred_list = []
            mask_list = []
            total_val_loss = 0
            for data in tqdm(data_loader, total=len(data_loader)):
                for k, v in data.items():
                    data[k] = v.to(device)

                with torch.no_grad():
                    val_logits, val_loss = model(**data)
                    # pred_tags: (nbest, batch_size, seq_length)
                    pred_tags = model.crf.decode(val_logits, data['attention_mask'])

                total_val_loss += val_loss.item()

                y_true_list.append(data["label_ids"])
                y_pred_list.append(pred_tags)
                mask_list.append(data["attention_mask"])

            y_true_stack = torch.stack(y_true_list)
            y_pred_stack = torch.stack(y_pred_list)
            mask_stack = torch.stack(mask_list)


            f1_score = f1_score_func(y_true=data['label_ids'],
                                         y_pred=pred_tags,
                                         mask=data['attention_mask'])




            print(f"Epochs: {epoch + 1}/{ner_config.EPOCHS}")
            print(f"Step: {steps}")




    return total_train_loss / len(data_loader), steps


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





train_dataset = NerDataset(file_name=FILE_NAME, mode="train")
dev_dataset = NerDataset(file_name=FILE_NAME, mode="dev")
test_dataset = NerDataset(file_name=FILE_NAME, mode="test")


train_dataloader = DataLoader(train_dataset, batch_size=ner_config.TRAIN_BATCH_SIZE,
                              shuffle=True, drop_last=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=ner_config.DEV_BATCH_SIZE,
                            shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=ner_config.TEST_BATCH_SIZE,
                            shuffle=True, drop_last=True)



