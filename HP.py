BASE_METADATA = "/mnt/storage1/Data/gutenberg/API/metadata/base_metadata.json"
BOOKS_METADATA = "/mnt/storage1/Data/gutenberg/API/metadata/books_metadata.json"
FEATURES_METADATA = "/mnt/storage1/Data/gutenberg/API/metadata/features_metadata.json"
BOOKS_DIR = "/mnt/storage1/Data/gutenberg/books/"
PARAGRAPH_METADATA = "/mnt/storage1/Data/gutenberg/API/metadata/paragraphs.npy"
TRAIN_PATH = "/mnt/storage1/Data/last-sentence-data/train"
TEST_PATH = "/mnt/storage1/Data/last-sentence-data/test"
DATA_PATH = "/mnt/storage1/Data/last-sentence-data/small"

pre_keys = ["is_analysed", "sents_num", "words_num", "tokens_num", "has_dialogue", "whole_dialogue"]
keys = ["global_id", "book_id", "local_id"] + pre_keys

