import kenlm
import math
from typing import List, Tuple

class KenLMPredictor:
    def __init__(self, model_path: str, vocab_path: str = None, vocab_list: List[str] = None):
        self.model = kenlm.Model(model_path)
        self.order = self.model.order

        # Załaduj słownik z pliku lub listy
        if vocab_path:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = [line.strip() for line in f if line.strip()]
        elif vocab_list:
            self.vocab = [w for w in vocab_list if not w.startswith("<")]
        else:
            raise ValueError("Musisz podać vocab_path lub vocab_list")

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
    MODEL_PATH = "model_3gram.binary"
    VOCAB_PATH = "vocab.txt"  # plik ze słowami, jedno słowo w linii

    predictor = KenLMPredictor(MODEL_PATH, vocab_path=VOCAB_PATH)

    context = "chciałabym powiedzieć, że choć przedstawienie było wielce interesujące, to nie było na "
    top_k = 5

    predictions = predictor.predict_next(context, k=top_k)

    print(f'Kontekst: "{context}"')
    print("Najbardziej prawdopodobne kolejne słowa:\n")

    for word, logp in predictions:
        prob = 10 ** logp
        print(f"{word:20s} logP={logp:.4f}  P≈{prob:.6f}")