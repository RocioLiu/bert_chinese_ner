import numpy as np
from tqdm import tqdm

import torch
from torch import nn
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



MODEL_CLASSES = {
    'bert': (BertConfig, BertCrfForNer, ner_config.TOKENIZER),
    # 'albert': (AlbertConfig, AlbertSoftmaxForNer, ner_config.TOKENIZER),
}



def training_fn(train_dataloader, dev_dataloader,
                model, optimizer, scheduler,
                epoch, device, steps):

    model.train()
    total_train_loss = 0
    train_losses = []

    # train_progress_bar = tqdm(train_dataloader, desc=f"Epoch: {epoch}",
    #                           leave=False, disable=False)

    # get a batch of data dict
    for data in tqdm(train_dataloader):

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

        train_losses.append(train_loss)
        total_train_loss += train_loss.item()

        if steps % ner_config.EVERY_N_STEP == 0:

            model.eval()
            y_true_list = []
            y_pred_list = []
            mask_list = []

            total_eval_loss = 0
            eval_losses = []

            for data in dev_dataloader:
                for k, v in data.items():
                    data[k] = v.to(device)

                with torch.no_grad():
                    eval_logits, eval_loss = model(**data)
                    # pred_tags: (nbest, batch_size, seq_length)
                    pred_tags = model.crf.decode(eval_logits, data['attention_mask'])

                eval_losses.append(eval_loss)
                total_eval_loss += eval_loss.item()

                y_true_list.append(data["label_ids"])
                y_pred_list.append(pred_tags)
                mask_list.append(data["attention_mask"])

            y_true_stack = torch.stack(y_true_list)
            y_pred_stack = torch.stack(y_pred_list)
            mask_stack = torch.stack(mask_list)

            avg_train_loss = total_train_loss / ner_config.EVERY_N_STEP
            avg_eval_loss = total_eval_loss / ner_config.EVERY_N_STEP

            # f1 score of a dev_dataloader
            eval_f1 = f1_score_func(y_true=y_true_stack,
                                    y_pred=y_pred_stack,
                                    mask=mask_stack)

            print(f"\nEpochs: {epoch}/{ner_config.EPOCHS}")
            print(f"Step: {steps}")
            print(f"Train loss: {avg_train_loss:.4f}")
            print(f"Eval loss: {avg_eval_loss:.4f}")
            print(f"Eval F1-score: {eval_f1:.4f} \n")

    # return the last eval_f1 after traverse an epoch
    return steps, train_losses, eval_losses, eval_f1



# here!! predict a new data
def predict_fn(data_loader, model):
    model.eval()
    total__loss = 0

    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)

        with torch.no_grad():
            loss, outputs = model(**data)

        total_loss += loss.item() # has loss or not

    return


# ---

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    FILE_NAME = {"train": ner_config.TRAINING_FILE, "dev": ner_config.DEV_FILE,
                 "test": ner_config.TEST_FILE}

    processor = CNerProcessor()

    label_list = processor.get_labels()
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for i, label in enumerate(label_list)}

    train_dataset = NerDataset(file_name=FILE_NAME, mode="train")
    dev_dataset = NerDataset(file_name=FILE_NAME, mode="dev")
    test_dataset = NerDataset(file_name=FILE_NAME, mode="test")

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=ner_config.TRAIN_BATCH_SIZE,
                                  shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=ner_config.DEV_BATCH_SIZE,
                                shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=ner_config.TEST_BATCH_SIZE,
                                 shuffle=True, drop_last=True)

    # --
    model_config, model_class, model_tokenizer = MODEL_CLASSES[ner_config.PRETRAINED_MODEL_NAME]

    model = model_class(pretrained_model_name=ner_config.BASE_MODEL_NAME,
                        config=BertConfig.from_pretrained(
                            ner_config.BASE_MODEL_NAME),
                        num_tags=len(ner_config.LABELS),
                        batch_first=True).to(device)
    # model.to(device)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())

    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_param_optimizer if
                    not any(nd in n for nd in no_decay)],
         "weight_decay": ner_config.WEIGHT_DECAY,
         "lr": ner_config.LEARNING_RATE},
        {"params": [p for n, p in bert_param_optimizer if
                    any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, "lr": ner_config.LEARNING_RATE},

        {"params": [p for n, p in crf_param_optimizer if
                    not any(nd in n for nd in no_decay)],
         "weight_decay": ner_config.WEIGHT_DECAY,
         "lr": ner_config.CRF_LEARNING_RATE},
        {"params": [p for n, p in crf_param_optimizer if
                    any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, "lr": ner_config.CRF_LEARNING_RATE},

        {"params": [p for n, p in linear_param_optimizer if
                    not any(nd in n for nd in no_decay)],
         "weight_decay": ner_config.WEIGHT_DECAY,
         "lr": ner_config.CRF_LEARNING_RATE},
        {"params": [p for n, p in linear_param_optimizer if
                    any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, "lr": ner_config.CRF_LEARNING_RATE},
    ]

    NUM_TRAIN_STEPS = int(len(train_dataloader) * ner_config.EPOCHS)

    WARMUP_STEPS = int(NUM_TRAIN_STEPS * ner_config.WARMUP_PROPORTION)
    optimizer = AdamW(optimizer_grouped_parameters, lr=ner_config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS,
        num_training_steps=NUM_TRAIN_STEPS
    )


    # -- training process --
    best_f1 = 0
    steps = 0
    for epoch in tqdm(range(1, ner_config.EPOCHS + 1)):

        total_train_losses = []
        total_eval_losses = []
        total_eval_f1 = []

        steps, train_losses, eval_losses, eval_f1 = training_fn(train_dataloader,
                                                                dev_dataloader,
                                                                model, optimizer,
                                                                scheduler, epoch,
                                                                device, steps)
        total_train_losses.append(train_losses)
        total_eval_losses.append(eval_losses)

        if eval_f1 > best_f1:
            torch.save(model.state_dict(), ner_config.MODEL_PATH)
            best_f1 = eval_f1


    # test on test_dataset!!