import kenlm
import math
from typing import List, Tuple

class KenLMPredictor:
    def __init__(self, model_name: str, vocab_size: int):
        self.model = kenlm.Model(model_name + ".binary")
        self.order = self.model.order
        arpa_path = model_name + ".arpa"
        self.vocab = self._load_unigram_vocab(arpa_path, vocab_size)

    @staticmethod
    def _load_unigram_vocab(arpa_path: str, max_vocab: int) -> List[str]:
        vocab = []
        with open(arpa_path, "r", encoding="utf8") as f:
            in_1gram = False
            for line in f:
                line = line.strip()
                if line == "\\1-grams:":
                    in_1gram = True
                    continue
                if in_1gram:
                    if line.startswith("\\"):
                        break
                    parts = line.split()
                    if len(parts) >= 2:
                        logprob = float(parts[0])
                        word = parts[1]
                        vocab.append((logprob, word))

        # sortuj wg częstości (logprob ~ częstotliwość)
        vocab.sort(reverse=True)
        vocab = [w for _, w in vocab[:max_vocab]]
        return vocab

    @staticmethod
    def _split_context_and_prefix(text: str) -> Tuple[List[str], str]:
        """
        Zwraca:
        - listę skończonych tokenów (kontekst)
        - prefiks rozpoczętego słowa ("" jeśli brak)
        """

        if not text or text.endswith(" "):
            return text.split(), ""

        context_tokens = text.split()

        return context_tokens[:-1], context_tokens[-1]

    def predict_next(
            self,
            text: str,
            k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Zwraca k najlepszych predykcji:
        - pełne słowa (jeśli brak prefiksu)
        - lub uzupełnienia prefiksu
        """

        context_tokens, prefix = self._split_context_and_prefix(text)
        print(context_tokens, prefix)

        history = context_tokens[-(self.order - 1):]

        base_score = self.model.score(
            " ".join(history),
            bos=False,
            eos=False
        )

        candidates = []

        for word in self.vocab:
            if prefix and not word.startswith(prefix):
                continue

            sentence = " ".join(history + [word])

            score = self.model.score(sentence, bos=False, eos=False)
            delta = score - base_score

            candidates.append((word, delta))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

if __name__ == "__main__":
    MODEL_NAME = "model_3gram"
    VOCAB_SIZE = 200_000

    predictor = KenLMPredictor(MODEL_NAME, VOCAB_SIZE)
    print(predictor.vocab[:10])

    context = "chociaż mam prawie trzydzieści lat cały czas czuję się "
    top_k = 5

    predictions = predictor.predict_next(context, k=top_k)

    print(f'Kontekst: "{context}"')
    print("Najbardziej prawdopodobne kolejne słowa:\n")

    for word, logp in predictions:
        prob = 10 ** logp
        print(f"{word:20s} logP={logp:.4f}  P≈{prob:.6f}")