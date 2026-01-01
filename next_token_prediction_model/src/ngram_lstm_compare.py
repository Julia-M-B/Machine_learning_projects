import random

import numpy as np
from typing import List, Tuple, Dict
import json
from abc import ABC, abstractmethod
import time
from collections import deque


class NextTokenPredictor(ABC):
    @abstractmethod
    def predict(self, text: str, k: int = 10) -> List[Tuple[str, float]]:
        """Zwraca [(token, probability), ...]"""
        pass

    def get_probability(self, text: str, next_token: str) -> float:
        """Zwraca prawdopodobieństwo konkretnego tokenu"""
        pass


class NGramPredictor(NextTokenPredictor):
    def __init__(self, model_path: str, vocab_path: str):
        import kenlm
        self.model = kenlm.Model(model_path)
        self.order = self.model.order
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = [line.strip() for line in f if line.strip() and not line.startswith("<")]

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

    def predict(self, text: str, k: int = 10) -> List[Tuple[str, float]]:
        context_tokens, prefix = self._split_context_and_prefix(text)

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
        # Konwertuj log-prob na prawdopodobieństwa (w kenlm korzysta się z log_10)
        return [(word, 10 ** logp) for word, logp in candidates[:k]]

    def get_probability(self, text: str, next_token: str) -> float:
        tokens = text.strip().split()
        history = tokens[-(self.order - 1):]
        base_score = self.model.score(" ".join(history), bos=False, eos=False)
        sentence = " ".join(history + [next_token])
        score = self.model.score(sentence, bos=False, eos=False)
        return 10 ** (score - base_score)


class LSTMPredictor(NextTokenPredictor):
    def __init__(self, model_dir: str, beam_width: int = 50, max_word_length: int = 10, device: str = None):
        from beam_search import create_beam_searcher
        import torch

        self.device = torch.device(device if device else "cpu")
        self.searcher = create_beam_searcher(model_dir=model_dir,
                                             beam_width=beam_width,
                                             max_word_length=max_word_length,
                                             device=device
                                             )


    def predict(self, context: str, k: int = 10) -> List[Tuple[str, float]]:
        top_words = self.searcher.get_top_k_words(context_text=context, k=k)
        predictions = [(word, prob) for word, prob, _ in top_words]
        return predictions

    def get_probability(self, context: str, next_token: str) -> float:
        prob = self.searcher.get_word_probability(context, next_token)
        return prob


class ModelEvaluator:
    def __init__(self, models: Dict[str, NextTokenPredictor], seq_len: int = 50):
        self.models = models
        self.seq_len = seq_len
        self.results = {name: {
            'accuracy@1': [],
            'accuracy@5': [],
            'accuracy@10': [],
            'mrr': [],
            'perplexity': [],
            'inference_time': [],
            "n_preds": [],
        } for name in models.keys()}

    def context_target_pairs_generator(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        words = text.split()
        # print(len(words))
        sequence = []
        for word in words:
            sequence.append(word)
            while len(sequence) >= self.seq_len + 1:
                # print(sequence)
                yield " ".join(sequence[:self.seq_len]) + " ", sequence[self.seq_len]
                sequence = sequence[int(self.seq_len / 2):]


    def evaluate_file(self, file_path: str):
        preds_counter = 0
        for context, target in self.context_target_pairs_generator(file_path):
            preds_counter += 1
            for model_name, model in self.models.items():
                start_time = time.time()

                predictions = model.predict(context, k=10)
                if not predictions:
                    continue

                pred_tokens = [token for token, _ in predictions]

                inference_time = time.time() - start_time

                # Metryki
                acc1 = 1.0 if target == pred_tokens[0] else 0.0
                acc5 = 1.0 if target in pred_tokens[:5] else 0.0
                acc10 = 1.0 if target in pred_tokens[:10] else 0.0
                mrr = self._calculate_mrr(pred_tokens, target)

                # Perplexity = exp(cross-entropy) = exp(średnie neg-log-prob)
                # log-prob -> wartość na minusie, bo prob [0, 1]
                # neg-log-prob = - log-prob ([0, +inf])
                # uwaga: exp jest wtedy, gdy użyty logarytm to ln
                # w przypadku log_10, ppl = 10 ** cross-entropy
                prob = model.get_probability(context, target)
                log_prob = np.log(prob + 1e-10)  # unikaj log(0)

                # Zapisz wyniki
                self.results[model_name]['accuracy@1'].append(acc1)
                self.results[model_name]['accuracy@5'].append(acc5)
                self.results[model_name]['accuracy@10'].append(acc10)
                self.results[model_name]['mrr'].append(mrr)
                self.results[model_name]['perplexity'].append(log_prob)
                self.results[model_name]['inference_time'].append(
                    inference_time)

        for model_name in self.models.keys():
            self.results[model_name]["n_preds"].append(preds_counter)


    @staticmethod
    def get_files_paths(indices, paths_file, offsets_file):
        from model import read_by_index
        return read_by_index(indices.tolist(), paths_file,
                                offsets_file)

    def evaluate_directory(self):
        from tqdm import tqdm
        import yaml
        import random


        with open("next_token_prediction_model/config.yml") as f:
            config = yaml.safe_load(f)

        random.seed(config["seed"])
        np.random.seed(config["seed"])

        test_indices = np.load(config["test-indices-file"])
        paths_file = config["train-paths-file"]
        offsets_file = config["train-offsets-file"]

        files = self.get_files_paths(test_indices, paths_file, offsets_file)
        random.shuffle(files)
        files = files[:10]

        for file in tqdm(files, desc="Evaluating files"):
            self.evaluate_file(file)

    def _calculate_mrr(self, predictions: List[str], true_token: str) -> float:
        try:
            rank = predictions.index(true_token) + 1
            return 1.0 / rank
        except ValueError:
            return 0.0

    def get_summary(self) -> Dict:
        """
        Zwraca podsumowanie wyników
        """
        summary = {}

        for model_name, metrics in self.results.items():
            summary[model_name] = {
                'accuracy@1': np.mean(metrics['accuracy@1']),
                'accuracy@5': np.mean(metrics['accuracy@5']),
                'accuracy@10': np.mean(metrics['accuracy@10']),
                'mrr': np.mean(metrics['mrr']),
                'perplexity': np.exp(-np.mean(metrics['perplexity'])),
                'avg_inference_time_ms': np.mean(
                    metrics['inference_time']) * 1000,
                'model_predictions': len(metrics['accuracy@1']),
                'total_predictions': sum(metrics['n_preds'])
            }

        return summary

    def print_comparison(self):
        """
        Wyświetla porównanie w czytelnej formie
        """
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)

        for model_name, metrics in summary.items():
            print(f"\n{model_name.upper()}:")
            print(
                f"  Accuracy@1:  {metrics['accuracy@1']:.4f} ({metrics['accuracy@1'] * 100:.2f}%)")
            print(
                f"  Accuracy@5:  {metrics['accuracy@5']:.4f} ({metrics['accuracy@5'] * 100:.2f}%)")
            print(
                f"  Accuracy@10: {metrics['accuracy@10']:.4f} ({metrics['accuracy@10'] * 100:.2f}%)")
            print(f"  MRR:         {metrics['mrr']:.4f}")
            print(f"  Perplexity:  {metrics['perplexity']:.2f}")
            print(f"  Avg Time:    {metrics['avg_inference_time_ms']:.2f} ms")
            print(f"  Model Preds: {metrics['model_predictions']}")
            print(f"  Total Preds: {metrics['total_predictions']}")

        # Porównanie bezpośrednie
        print("\n" + "-" * 80)
        print("WINNER IN EACH CATEGORY:")
        print("-" * 80)

        categories = ['accuracy@1', 'accuracy@5', 'accuracy@10', 'mrr']
        for cat in categories:
            best_model = max(summary.items(), key=lambda x: x[1][cat])
            print(f"  {cat:15s}: {best_model[0]} ({best_model[1][cat]:.4f})")

        # Perplexity - niższe jest lepsze
        best_perplexity = min(summary.items(), key=lambda x: x[1]['perplexity'])
        print(
            f"  {'perplexity':15s}: {best_perplexity[0]} ({best_perplexity[1]['perplexity']:.2f})")

        # Czas - szybszy jest lepszy
        best_time = min(summary.items(),
                        key=lambda x: x[1]['avg_inference_time_ms'])
        print(
            f"  {'speed':15s}: {best_time[0]} ({best_time[1]['avg_inference_time_ms']:.2f} ms)")

    def save_results(self, output_path: str):
        """
        Zapisuje szczegółowe wyniki do JSON
        """
        summary = self.get_summary()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nWyniki zapisane do: {output_path}")


if __name__ == "__main__":
    # Inicjalizuj modele
    ngram_model = NGramPredictor(
        model_path="next_token_prediction_model/ngram/model_3gram.binary",
        vocab_path="next_token_prediction_model/ngram/vocab.txt"
    )

    lstm_model = LSTMPredictor(
        model_dir="next_token_prediction_model/",
        beam_width=50,
        max_word_length=10,
        device="cpu"
    )

    # Stwórz evaluator
    models = {
        'n-gram': ngram_model,
        'lstm': lstm_model
    }

    evaluator = ModelEvaluator(models)

    # Uruchom ewaluację
    print("Rozpoczynam ewaluację...")
    evaluator.evaluate_directory()

    # Wyświetl wyniki
    evaluator.print_comparison()

    # Zapisz do pliku
    evaluator.save_results("comparison_results.json")