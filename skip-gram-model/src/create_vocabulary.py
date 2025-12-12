import sys

from tqdm import tqdm
from utils import CONFIG_PATH
import yaml
import re
import glob
from collections import Counter
import pickle

word_regex = re.compile("\S+")
letters_only_regex = re.compile("[^a-ząćęłńóśźż\s]")
multiple_whitespaces_regex = re.compile("\s+")


class Vocab:
    def __init__(self):
        self.words_counter = Counter()
        self.word2idx: dict[str, int] = dict()
        self.idx2word: dict[int, str] = dict()

    def load_words_counter_from_file(self, file_path: str) -> None:
        with open(file_path, "rb") as f:
            self.words_counter = pickle.load(f)

    def save_words_counter_to_file(self, file_path: str) -> None:
        with open(file_path, "wb") as f:
            pickle.dump(self.words_counter, f)

    def update_words_counter(self, text) -> None:
        self.words_counter.update(text)

    def create_word_to_idx_mapping(self) -> None:
        if not self.words_counter:
            raise ValueError("Words counter is empty. Update words counter with text file(s) or load it from `.pkl` file.")

        if self.idx2word:
            word2idx = {v : k for k, v in self.idx2word.items()}
        else:
            word2idx: dict[str, int] = {"<unk>": 0}
            sorted_words_counter = dict(self.words_counter.most_common())
            for i, word in zip(range(1, len(sorted_words_counter)), sorted_words_counter.keys()):
                word2idx[word] = i

        self.word2idx = word2idx

    def create_idx_to_word_mapping(self) -> None:
        if not self.words_counter:
            raise ValueError("Words counter is empty. Update words counter with text file(s) or load it from `.pkl` file.")

        if self.word2idx:
            idx2word = {v : k for k, v in self.word2idx.items()}
        else:
            idx2word: dict[int, str] = {0: "<unk>"}
            sorted_words_counter = dict(self.words_counter.most_common())
            for i, word in zip(range(1, len(sorted_words_counter)), sorted_words_counter.keys()):
                idx2word[i] = word

        self.idx2word = idx2word

    def get_idx_for_word(self, word: str) -> int:
        idx = self.word2idx.get(word)
        if idx:
            return idx
        else:
            return 0

    def get_word_for_idx(self, idx: int) -> str:
        word = self.idx2word.get(idx)
        if word:
            return word
        else:
            return self.idx2word.get(0)

def get_files_paths(base_path, datasets_names):
    """Finds all .txt files in a certain dir and yields one path at the time"""
    for dataset in datasets_names:
        yield from glob.glob(f"{base_path}/{dataset}/*.txt")

def clean(data: str) -> str:
    """Normalizes text by applying lowercasing, removing non-letters characters
    and removing multiple whitespaces"""
    data = data.strip().lower()
    data = letters_only_regex.sub("", data)
    data = multiple_whitespaces_regex.sub(" ", data)
    return data

def create_vocabulary_from_file(file_path: str, vocab: Vocab):
    with open(file_path) as f:
        data = clean(f.read()).split(" ")  #
        vocab.update_words_counter(data)


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    datasets_names = config["data"]["datasets"]

    lemmatize = config["lemmatization"]["lemmatize"]

    if lemmatize:
        database_path = config["data"]["lemmatized-data-dir"]
    else:
        database_path = config["data"]["data-dir"]

    vocab = Vocab()
    words_counter_name = config["vocabulary"]["words-counter-name"]

    n_files = len(list(get_files_paths(database_path, datasets_names)))
    files_paths = get_files_paths(database_path, datasets_names)

    for file_path in tqdm(files_paths, total=n_files):
        create_vocabulary_from_file(file_path, vocab)

    vocab.save_words_counter_to_file(words_counter_name)

    print("Words counter size in bytes:", sys.getsizeof(vocab.words_counter))

    n = 50
    print(f"Words counter first {n} items:")
    for _, item in zip(range(n), vocab.words_counter.items()):
        print(item)


if __name__ == "__main__":
    main()
