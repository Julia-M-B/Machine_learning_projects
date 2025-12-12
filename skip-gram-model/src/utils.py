import re
import numpy as np

CONFIG_PATH = "../config.yaml"
WORD_REGEX = re.compile("\S+")

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


def get_single_words(data):
    """
    generator;
    wynajduje stringi, które nie są ciągiem (dowolnie długim) białych znaków,
    a następnie zwraca kolejne obiekty, które w pełni pasują do wyrażenia
    (pełna kompatybilność: argument `0` w metodzie `group`)
    """
    for match in WORD_REGEX.finditer(data):
        yield match.group(0)


