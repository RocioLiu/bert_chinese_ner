import numpy as np
import pandas as pd
from os.path import join
import json
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
from bert_chinese_ner.utils.plot_loss import loss_f1_plot
from bert_chinese_ner.models.transformers.models.bert.configuration_bert import BertConfig


MODEL_CLASSES = {
    'bert': (BertConfig, BertCrfForNer, ner_config.TOKENIZER),
    # 'albert': (AlbertConfig, AlbertForNer, ner_config.TOKENIZER),
}


def training_fn(train_dataloader, dev_dataloader,
                model, optimizer, scheduler, epoch, device, steps,
                history_dict, history_df):

    model.train()
    total_train_loss = 0 # reset every epoch
    # train_losses = []
    dev_steps = len(dev_dataloader)

    train_progress_bar = tqdm(train_dataloader, desc=f"Epoch: {epoch}",
                              leave=False, disable=False)

    # get a batch of data dict
    for data in train_progress_bar:

        steps += 1
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        train_logits, train_loss = model(**data)
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), ner_config.GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        # train_losses.append(train_loss)
        # total_train_loss += train_loss.item()


        if steps % ner_config.EVERY_N_STEP == 0:

            model.eval()
            y_true_list = []
            y_pred_list = []
            mask_list = []

            total_eval_loss = 0
            # eval_losses = []

            for data in dev_dataloader:
                for k, v in data.items():
                    data[k] = v.to(device)

                with torch.no_grad():
                    eval_logits, eval_loss = model(**data)
                    # pred_tags: (nbest, batch_size, seq_length)
                    pred_tags = model.crf.decode(eval_logits, data['attention_mask'])

                # eval_losses.append(eval_loss)
                total_eval_loss += eval_loss.item()

                y_true_list.append(data["label_ids"])
                y_pred_list.append(pred_tags)
                mask_list.append(data["attention_mask"])

            y_true_stack = torch.stack(y_true_list)
            y_pred_stack = torch.stack(y_pred_list)
            mask_stack = torch.stack(mask_list)

            # avg_train_loss = total_train_loss / steps
            avg_eval_loss = total_eval_loss / dev_steps


            # f1 score of a dev_dataloader
            eval_f1 = f1_score_func(y_true=y_true_stack,
                                    y_pred=y_pred_stack,
                                    mask=mask_stack)

            history_dict['step'].append(steps)
            history_dict['train_loss'].append(avg_train_loss)
            history_dict['eval_loss'].append(avg_eval_loss)
            history_dict['eval_f1'].append(eval_f1)

            history_row = pd.DataFrame([[steps, avg_train_loss, avg_eval_loss, eval_f1]],
                                      columns=['step', 'train_loss', 'eval_loss', 'eval_f1'])
            history_df = pd.concat([history_df, history_row], ignore_index=True, axis=0)

            print(f"\nEpoch: {epoch}/{ner_config.EPOCHS}    step: {steps}")
            # print(f"Step: {steps}")
            print(f"train_loss: {train_loss.item():.4f} - eval_loss: {avg_eval_loss:.4f} - eval_f1: {eval_f1:.4f} \n")
            # print(f"Eval loss: {avg_eval_loss:.4f}")
            # print(f"Eval F1-score: {eval_f1:.4f} \n")

    # return the last eval_f1 after traverse an epoch
    return steps, eval_f1, history_dict, history_df


def eval_fn(data_loader, model, device):
    model.eval()
    y_true_list = []
    y_pred_list = []
    mask_list = []

    for data in data_loader:
        for k, v in data.items():
            data[k] = v.to(device)

        with torch.no_grad():
            eval_logits, eval_loss = model(**data)
            pred_tags = model.crf.decode(eval_logits, data['attention_mask'])

        y_true_list.append(data["label_ids"])
        y_pred_list.append(pred_tags)
        mask_list.append(data["attention_mask"])

    y_true_stack = torch.stack(y_true_list)
    y_pred_stack = torch.stack(y_pred_list)
    mask_stack = torch.stack(mask_list)

    # f1 score of a test_dataloader
    eval_f1 = f1_score_func(y_true=y_true_stack,
                            y_pred=y_pred_stack,
                            mask=mask_stack)

    return eval_f1


def predict_fn(inputs, model, id_to_label, model_tokenizer, device):

    is_sentence = True if type(inputs) == str else False

    if is_sentence:
        tokenized_sent = model_tokenizer.tokenize(inputs)
        data = model_tokenizer.encode_plus(inputs,
                                            max_length=ner_config.MAX_LEN,
                                            padding='max_length',
                                            return_attention_mask=True,
                                            return_tensors='pt')

        with torch.no_grad():
            for k, v in data.items():
                data[k] = v.to(device)

            logits = model(**data)

        pred_tags = model.crf.decode(logits, data['attention_mask']).cpu().numpy().reshape(-1)
        y_pred = [id_to_label[i] for i in pred_tags[1:data['attention_mask'].sum() - 1]]

        for c, p in zip(tokenized_sent, y_pred):
            print(c, p)

    else:
        tokenized_sent = model_tokenizer.tokenize(model_tokenizer.decode(
            inputs['input_ids'], skip_special_tokens=True))
        data = inputs

        with torch.no_grad():
            for k, v in data.items():
                data[k] = v.unsqueeze(0).to(device)

            logits, loss = model(**data)

        pred_tags = model.crf.decode(logits, data['attention_mask']).cpu().numpy().reshape(-1)
        y_pred = [id_to_label[i] for i in pred_tags[1:data['attention_mask'].sum()-1]]
        y_true = [id_to_label[i] for i in data['label_ids'].squeeze(0).numpy()[1:data['attention_mask'].sum()-1]]

        for c, p, t in zip(tokenized_sent, y_pred, y_true):
            print(c, p, t)


# ---

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    FILE_NAME = {"train": ner_config.TRAINING_FILE, "dev": ner_config.DEV_FILE,
                 "test": ner_config.TEST_FILE}

    processor = CNerProcessor()

    label_list = processor.get_labels()
    label_to_id = processor.get_label_to_id()
    id_to_label = processor.get_id_to_label()

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
                        config=BertConfig.from_pretrained(ner_config.BASE_MODEL_NAME),
                        num_tags=len(label_list),
                        batch_first=True)
    model.to(device)

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
    history_df = pd.DataFrame(columns=['step', 'train_loss', 'eval_loss', 'eval_f1'])
    history_dict = {"step": [],
                    "train_loss": [],
                    "eval_loss": [],
                    "eval_f1": []}

    for epoch in tqdm(range(1, ner_config.EPOCHS + 1)):
        steps, eval_f1, history_dict, history_df = training_fn(train_dataloader,
                                                               dev_dataloader,
                                                               model, optimizer,
                                                               scheduler, epoch,
                                                               device, steps,
                                                               history_dict,
                                                               history_df)
        # total_train_losses.append(train_losses)
        # total_eval_losses.append(eval_losses)

        if eval_f1 > best_f1:
            torch.save(model.state_dict(), ner_config.MODEL_PATH)
            best_f1 = eval_f1

    with open(ner_config.OUTPUT_JSON, 'w') as file:
        json.dump(history_dict, file)

    history_df.to_csv(ner_config.OUTPUT_CSV)


    # --
    # Load the trained model for evaluating on test dataset and prediction
    model = model_class(pretrained_model_name=ner_config.BASE_MODEL_NAME,
                        config=BertConfig.from_pretrained(ner_config.BASE_MODEL_NAME),
                        num_tags=len(label_list),
                        batch_first=True)
    # , map_location=torch.device('cpu')
    state_dict = torch.load(ner_config.MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.to(device)


    # evaluate the model on test_dataloader
    test_f1 = eval_fn(test_dataloader, model, device)
    print(f"\n\nf1 score on test dataset: {test_f1}\n\n")


    # Prediction on an example
    inputs = ner_config.TEST_SENTENCE
    predict_fn(inputs, model, id_to_label,model_tokenizer, device)

    # Prediction on an of test_dataset
    inputs = test_dataset[121]
    predict_fn(inputs, model, id_to_label, model_tokenizer, device)

    # load history.json
    with open(ner_config.OUTPUT_JSON, 'r') as json_file:
        history = json.load(json_file)


    # plot the losses and f1-scores
    loss_f1_plot(history, ner_config.EPOCHS, ner_config.IMG_PATH,
                 plot_title='Bert + CRF layer - Performance',
                 y_label='CRF loss')



if __name__ == "__main__":
    main()
