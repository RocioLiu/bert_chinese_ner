# import transformers
from os.path import abspath, dirname, split, join

# from bert_chinese_ner.models.transformers.models.bert.tokenization_bert import BertTokenizer
from .models.transformers.models.bert.tokenization_bert import BertTokenizer


ROOT_DIR = join(*split(abspath(dirname(__file__)))[:-1])
#ROOT_DIR = abspath(dirname(__file__))

DATA_DIR = join(ROOT_DIR, "data")
MODEL_PATH = join(ROOT_DIR, "outputs")

TRAINING_FILE = "example.train"
DEV_FILE = "example.dev"
TEST_FILE = "example.test"

PRETRAINED_MODEL_NAME = 'bert'

EPOCHS = 3
PRINT_EVERY_N_STEP = 50

TRAIN_BATCH_SIZE = 64
DEV_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

MAX_LEN = 256

BASE_MODEL_NAME = "bert-base-chinese"
VOCAB_FILE = join("data", BASE_MODEL_NAME, "vocab.txt")
LABELS = ['B-LOC', 'B-ORG', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER', 'O']


TOKENIZER = BertTokenizer.from_pretrained(
    BASE_MODEL_NAME,
    do_lower_case=False
)

WEIGHT_DECAY = 0.01
LEARNING_RATE = 3e-5
CRF_LEARNING_RATE = 3e-5
WARMUP_PROPORTION = 0.1

GRAD_CLIP = 1.0
