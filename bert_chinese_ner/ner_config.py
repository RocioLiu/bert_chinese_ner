# import transformers
from os.path import abspath, dirname, split, join

# from bert_chinese_ner.models.transformers.models.bert.tokenization_bert import BertTokenizer
from .models.transformers.models.bert.tokenization_bert import BertTokenizer


ROOT_DIR = join(*split(abspath(dirname(__file__)))[:-1])
#ROOT_DIR = abspath(dirname(__file__))

DATA_DIR = join(ROOT_DIR, "data")
OUTPUT_PATH = join(ROOT_DIR, "outputs")
MODEL_PATH = join(OUTPUT_PATH, "checkpoint", "model.pth")

TRAINING_FILE = "example.train"
DEV_FILE = "example.dev"
TEST_FILE = "example.test"

PRETRAINED_MODEL_NAME = 'bert'

EPOCHS = 1
EVERY_N_STEP = 5

TRAIN_BATCH_SIZE = 64
DEV_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

MAX_LEN = 256

BASE_MODEL_NAME = "bert-base-chinese"
VOCAB_FILE = join("data", BASE_MODEL_NAME, "vocab.txt")
# LABELS = ['B-LOC', 'B-ORG', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER', 'O']


TOKENIZER = BertTokenizer.from_pretrained(
    BASE_MODEL_NAME,
    do_lower_case=False
)

WEIGHT_DECAY = 0.01
LEARNING_RATE = 3e-5
CRF_LEARNING_RATE = 3e-5
WARMUP_PROPORTION = 0.1
GRAD_CLIP = 1.0

TEST_SENTENCE = "我想要去南港citylink的麥當勞用優惠券買勁辣雞腿堡買一送一，結果遇到台南金城武。"
