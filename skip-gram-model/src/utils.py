import pickle
import glob
import re
from collections import Counter
import numpy as np

CONFIG_PATH = "../config.yaml"
WORD_REGEX = re.compile("\S+")
MULTIPLE_WHITESPACE_REGEX = re.compile("\s+")
LEAVE_LETTERS_ONLY_REGEX = re.compile("[^a-ząćęłńóśźż\s]")


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


class SubsamplingHelper:
    def __init__(self, vocab: Vocab, p: int = 90):
        if not vocab.words_counter:
            raise ValueError("Trying to create SubsampledVocab with empty words counter. Update words counter first.")
        self.vocab = vocab
        self.threshold = self._calculate_threshold(p)
        self.vocab_size = len(self.vocab.words_counter)
        self.subsampling_probs = self._calculate_subsampling_probs()

    def _calculate_threshold(self, p) -> float:
        freqs = np.array(list(self.vocab.words_counter.values())) / sum(self.vocab.words_counter.values())
        return np.percentile(freqs, p)

    def _calculate_subsampling_probs(self) -> dict:
        """
        Calculate probability of keeping each word based on frequency.
        Formula from Mikolov et al.: P(w_i) = 1 - sqrt(t/f(w_i))
        where t is threshold and f(w_i) is frequency of word
        """
        probs = {}
        for word, count in self.vocab.words_counter.items():
            freq = count / self.vocab_size
            prob = max(0, 1.0 - np.sqrt(self.threshold / freq))  # Probability of discarding the word
            prob = 1 - prob  # Probability of keeping the word
            token = self.vocab.get_idx_for_word(word)
            probs[token] = prob
        return probs

    def subsample_word(self, token: int) -> [int, None]:
        if token in self.subsampling_probs:
            if np.random.random() < self.subsampling_probs[token]:  # Randomly keeping word with a given probability
                return token
            else:
                return
        else:
            return token


def get_files_paths(base_path, datasets_names):
    """
    generator ścieżek do plików tekstowych (zwraca jedną ścieżkę na raz)
    """
    for dataset in datasets_names:
        yield from glob.glob(f"{base_path}/{dataset}/*.txt")

def clean(data: str) -> str:
    """
    funkcja czyści dane tekstowe:
    - zamienia wielkie litery na małe;
    - usuwa wszystkie znaki, które nie są literami;
    """
    data = data.strip().lower()
    data = LEAVE_LETTERS_ONLY_REGEX.sub("", data)
    data = MULTIPLE_WHITESPACE_REGEX.sub(" ", data)
    return data

def get_single_words(data):
    """
    generator;
    wynajduje stringi, które nie są ciągiem (dowolnie długim) białych znaków,
    a następnie zwraca kolejne obiekty, które w pełni pasują do wyrażenia
    (pełna kompatybilność: argument `0` w metodzie `group`)
    """
    for match in WORD_REGEX.finditer(data):
        yield match.group(0)


