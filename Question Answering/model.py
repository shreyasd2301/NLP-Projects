import tokenizers
import os

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 10
BERT_PATH = "../input/bert_base_uncased/"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../inpu/train.csv"
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    os.path.join(BERT_PATH, "vocab.txt"),
    do_lower_case=True
)