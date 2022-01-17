#import transformers
from os.path import abspath, dirname, split, join

# REPO_DIR = join(*split(abspath(dirname(__file__)))[:-1])
ROOT_DIR = abspath(dirname(__file__))

TRAINING_FILE = "input/peoples_daily/example.train.txt"
DEV_FILE = "input/peoples_daily/example.dev.txt"
TEST_FILE = "input/peoples_daily/example.test.txt"

BASE_MODEL = "bert-base-chinese"

# TOKENIZER = transformers.BertTokenizer.from_pretrained(
#     BASE_MODEL,
#     do_lower_case=False
# )