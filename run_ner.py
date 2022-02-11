from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler


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
    # get a batch of data dict
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.transpose(0, 1).to(device)
        optimizer.zero_grad()
        loss, outputs = model(**data)
        loss.backward()
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

        loss, _ = model(**data)
        total_loss += loss.item()
    return total_loss / len(data_loader)





## hparams of BertCrfForNer class
config = BertConfig.from_pretrained(ner_config.BASE_MODEL_NAME)
num_tags = len(ner_config.LABELS)


tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None,)


FILE_NAME = {"train":ner_config.TRAINING_FILE, "dev":ner_config.DEV_FILE, "test":ner_config.TEST_FILE}


train_dataset = NerDataset(file_name=FILE_NAME, mode="train")
len(train_dataset) #20864

dev_dataset = NerDataset(file_name=FILE_NAME, mode="dev")
len(dev_dataset) #2318

test_dataset = NerDataset(file_name=FILE_NAME, mode="test")
len(test_dataset) #4636


# DistributedSampler(train_dataset)
# train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=ner_config.TRAIN_BATCH_SIZE,
                              shuffle=True, drop_last=True)

dev_dataloader = DataLoader(dev_dataset, batch_size=ner_config.DEV_BATCH_SIZE,
                            shuffle=True, drop_last=True)

test_dataloader = DataLoader(test_dataset, batch_size=ner_config.TEST_BATCH_SIZE,
                            shuffle=True, drop_last=True)





best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = train_fn(train_dataloader, model, optimizer, scheduler, device)
        dev_loss = eval_fn(dev_dataloader, model, device)
        print(f"Train Loss: {train_loss} Valid Loss: {valid_loss}")
        if dev_loss > best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = dev_loss