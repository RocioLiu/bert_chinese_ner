import transformers
from os.path import abspath, dirname, split, join

ROOT_DIR = join(*split(abspath(dirname(__file__)))[:-1])
#ROOT_DIR = abspath(dirname(__file__))

TRAINING_FILE = "data/example.train"
DEV_FILE = "data/example.dev"
TEST_FILE = "data/example.test"

BASE_MODEL = "bert-base-chinese"

# TOKENIZER = transformers.BertTokenizer.from_pretrained(
#     BASE_MODEL,
#     do_lower_case=False
# )