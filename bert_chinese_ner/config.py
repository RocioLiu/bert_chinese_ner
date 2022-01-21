import transformers
from os.path import abspath, dirname, split, join

ROOT_DIR = join(*split(abspath(dirname(__file__)))[:-1])
#ROOT_DIR = abspath(dirname(__file__))

DATA_DIR = join(ROOT_DIR, "data")
TRAINING_FILE = "example.train"
DEV_FILE = "example.dev"
TEST_FILE = "example.test"

BASE_MODEL_NAME = "bert-base-chinese"
VOCAB_FILE = join("data", BASE_MODEL_NAME, "vocab.txt")


MAX_LEN = 128

TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_NAME,
    do_lower_case=False
)