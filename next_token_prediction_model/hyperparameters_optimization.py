import glob
import json
import random
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import numpy as np
import sentencepiece as spm
import torch
from src.beam_search import WordPredictionBeamSearch
from src.model import (
    LSTMLanguageModel,
    StreamingTextDataset,
    compute_val_ppl,
    train_epoch,
)
from torch.utils.data import DataLoader


class LSTMModelWrapper:
    """
    Wrapper class for LSTM Language Model to provide prediction interface.

    This class encapsulates the LSTM model and provides a simplified interface
    for making predictions on token sequences.

    Attributes:
        device (str): Device to run the model on ('cuda' or 'cpu').
        model (LSTMLanguageModel): The underlying LSTM language model.
        vocab_size (int): Size of the vocabulary.
        seq_len (int): Maximum sequence length to consider for predictions.
    """

    def __init__(self, model: LSTMLanguageModel, seq_len: int = 64):
        """
        Initialize the LSTM model wrapper.

        Args:
            model: The LSTM language model to wrap.
            seq_len: Maximum sequence length for context. Defaults to 64.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.model.to(self.device)
        self.model.eval()

        self.vocab_size = self.model.vocab_size
        self.seq_len = seq_len

    def predict(self, context_tokens: List[int]) -> List[float]:
        """
        Predict probability distribution over next tokens given context.

        Args:
            context_tokens: List of token IDs representing the context.
                           If empty, returns uniform distribution.

        Returns:
            List of probabilities for each token in vocabulary.
            Length equals vocab_size.
        """
        if not context_tokens:
            return [1.0 / self.vocab_size] * self.vocab_size
        context_tokens = context_tokens[-self.seq_len :]
        input_ids = torch.LongTensor([context_tokens]).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(input_ids)
            last_logits = logits[0, -1, :]
            probs = torch.softmax(last_logits, dim=0)
            return probs.cpu().tolist()


@dataclass()
class Hyperparameters:
    """
    Data class storing all hyperparameters for LSTM model training.

    Attributes:
        tokenizer: Path to the SentencePiece tokenizer model file.
        emb_dim: Dimension of token embeddings.
        n_layers: Number of LSTM layers.
        hidden_units: Number of hidden units in LSTM layers.
        dropout: Dropout rate for regularization.
        weight_decay: L2 regularization coefficient.
        batch_size: Number of samples per training batch.
        seq_len: Length of input sequences.
        lr: Learning rate for optimizer.
    """

    tokenizer: str
    emb_dim: int
    n_layers: int
    hidden_units: int
    dropout: float
    weight_decay: float
    batch_size: int
    seq_len: int
    lr: float


@dataclass(order=True)
class ExperimentData:
    """
    Data class storing results from a single hyperparameter experiment.

    The class is ordered by best_score for easy comparison of experiments.

    Attributes:
        params: Dictionary of hyperparameter values used.
        best_score: Best validation accuracy achieved during training.
        history: Dictionary containing training metrics over epochs.
        trial_params: List of parameter names that were varied in this trial.
    """

    params: dict = field(compare=False)
    best_score: float
    history: dict = field(compare=False)
    trial_params: list = field(compare=False)


def get_top_k_accuracy(
    searcher: WordPredictionBeamSearch, context: str, true_word: str, k: int = 5
) -> float:
    """
    Compute top-k accuracy for a single prediction.

    Args:
        searcher: Beam search object for word prediction.
        context: String context to predict from.
        true_word: The actual next word to predict.
        k: Number of top predictions to consider. Defaults to 5.

    Returns:
        1.0 if true_word is in top-k predictions, 0.0 otherwise.
    """
    top_words = searcher.get_top_k_words(context_text=context, k=k)
    predictions = [word for word, _, _ in top_words]
    return 1.0 if true_word in predictions else 0.0


def evaluate_top_k_words(
    searcher: WordPredictionBeamSearch, files_paths: List[str], seq_len: int, k: int = 5
) -> float:
    """
    Evaluate top-k accuracy across multiple text files.

    Uses a sliding window approach to generate context-target pairs from
    the text files and computes the proportion where the true word appears
    in the top-k predictions.

    Args:
        searcher: Beam search object for word prediction.
        files_paths: List of paths to text files for evaluation.
        seq_len: Length of context window in words.
        k: Number of top predictions to consider. Defaults to 5.

    Returns:
        Average top-k accuracy across all predictions.
        Returns 0.0 if no predictions were made.
    """
    preds_counter = 0
    acc5 = 0
    slide = int(seq_len / 2)

    for file in files_paths:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        words = text.split()
        sequence = []

        for word in words:
            sequence.append(word)
            while len(sequence) >= seq_len + 1:
                context = " ".join(sequence[:seq_len]) + " "
                target = sequence[seq_len]
                acc5 += get_top_k_accuracy(searcher, context, target, k)
                preds_counter += 1
                sequence = sequence[slide:]

    return acc5 / preds_counter if preds_counter > 0 else 0


def create_and_train_model(
    train_files: List[str],
    val_files: List[str],
    hyperparameters: Hyperparameters,
    trial_params: List[str],
    n_epochs: int = 3,
    device: str = "cuda",
) -> ExperimentData:
    """
    Create, train and evaluate an LSTM language model.

    This function performs the complete training pipeline:
    1. Creates datasets and dataloaders
    2. Initializes the model and optimizer
    3. Trains for specified number of epochs
    4. Tracks training metrics and validation performance

    Args:
        train_files: List of file paths for training data.
        val_files: List of file paths for validation data.
        hyperparameters: Hyperparameter configuration for the model.
        trial_params: Names of hyperparameters being varied in grid search.
        n_epochs: Number of training epochs. Defaults to 3.
        device: Device to train on ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        ExperimentData object containing results, history and configuration.

    Note:
        The model is evaluated on validation set after each epoch.
        Best score is tracked based on validation top-5 accuracy.
    """
    train_ds = StreamingTextDataset(
        files=train_files,
        sp_model_path=hyperparameters.tokenizer,
        seq_len=hyperparameters.seq_len,
    )
    val_ds = StreamingTextDataset(
        files=val_files,
        sp_model_path=hyperparameters.tokenizer,
        seq_len=hyperparameters.seq_len,
    )

    train_dl = DataLoader(train_ds, batch_size=hyperparameters.batch_size)
    val_dl = DataLoader(val_ds, batch_size=hyperparameters.batch_size)

    sp_proc = spm.SentencePieceProcessor()
    sp_proc.load(hyperparameters.tokenizer)

    model = LSTMLanguageModel(
        vocab_size=sp_proc.get_piece_size(),
        emb_dim=hyperparameters.emb_dim,
        hidden_dim=hyperparameters.hidden_units,
        n_layers=hyperparameters.n_layers,
        dropout=hyperparameters.dropout,
    ).to(device)

    model_wrapper = LSTMModelWrapper(model=model, seq_len=hyperparameters.seq_len)

    searcher = WordPredictionBeamSearch(model=model_wrapper, tokenizer=sp_proc)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters.lr,
        weight_decay=hyperparameters.weight_decay,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_ppl": [],
        "val_acc5": [],
    }
    best_score = 0

    for epoch in range(1, n_epochs + 1):
        print(f"Training epoch {epoch}:")
        train_loss = train_epoch(model, train_dl, optimizer, device)
        print(f"Epoch {epoch} train loss: {train_loss:.4f}")

        val_loss, val_ppl = compute_val_ppl(model, val_dl, device)
        val_acc5 = evaluate_top_k_words(
            searcher, files_paths=val_files, seq_len=hyperparameters.seq_len
        )

        print(f"Epoch {epoch} val loss: {val_loss:.4f} ppl: {val_ppl:.2f}")
        print(f"Epoch {epoch} val acc: {val_acc5:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["val_acc5"].append(val_acc5)

        if best_score < val_acc5:
            best_score = val_acc5

    return ExperimentData(
        params=asdict(hyperparameters),
        best_score=best_score,
        history=history,
        trial_params=trial_params,
    )


def hyperparameters_setup_generator(
    fixed_config: Dict[str, any], trial_config: Dict[str, List[any]]
) -> Generator[Hyperparameters, None, None]:
    """
    Generate all combinations of hyperparameters for grid search.

    Creates Hyperparameters objects by combining fixed values with all
    possible combinations of trial values using Cartesian product.

    Args:
        fixed_config: Dictionary of fixed hyperparameter values.
        trial_config: Dictionary mapping parameter names to lists of values
                     to try in grid search.

    Yields:
        Hyperparameters object for each combination in the grid.

    Example:
        >>> fixed = {"n_layers": 3, "dropout": 0.2}
        >>> trial = {"lr": [1e-3, 1e-4], "batch_size": [32, 64]}
        >>> for hp in hyperparameters_setup_generator(fixed, trial):
        ...     # Will yield 4 combinations (2 lr × 2 batch_size)
        ...     print(hp.lr, hp.batch_size)
    """

    def _get_param_value(param_name, trial_params, trial_combination):
        """Helper to get parameter value from trial or fixed config."""
        if param_name in trial_params:
            idx = trial_params.index(param_name)
            return trial_combination[idx]
        fixed_val = fixed_config.get(param_name)
        return fixed_val

    trial_params_names = trial_config.keys()
    combinations = list(product(*trial_config.values()))

    for combination in combinations:
        yield Hyperparameters(
            tokenizer=_get_param_value("tokenizer", trial_params_names, combination),
            emb_dim=_get_param_value("emb_dim", trial_params_names, combination),
            n_layers=_get_param_value("n_layers", trial_params_names, combination),
            hidden_units=_get_param_value(
                "hidden_units", trial_params_names, combination
            ),
            dropout=_get_param_value("dropout", trial_params_names, combination),
            weight_decay=_get_param_value(
                "weight_decay", trial_params_names, combination
            ),
            batch_size=_get_param_value("batch_size", trial_params_names, combination),
            seq_len=_get_param_value("seq_len", trial_params_names, combination),
            lr=_get_param_value("lr", trial_params_names, combination),
        )


def run_grid_search(
    study_num: int,
    fixed_config: Dict[str, any],
    trial_config: Dict[str, List[any]],
    train_files: List[str],
    val_files: List[str],
    n_epochs: int = 3,
    device: str = "cuda",
    save_dir: str = "next_token_prediction_model/study_results",
) -> List[ExperimentData]:
    """
    Run grid search over hyperparameter combinations.

    Trains models for all combinations of trial parameters while keeping
    fixed parameters constant. Saves results to JSON files after each trial.

    Args:
        study_num: Identifier for this grid search study.
        fixed_config: Dictionary of hyperparameters to keep constant.
        trial_config: Dictionary of hyperparameters to vary, mapping names
                     to lists of values to try.
        train_files: List of file paths for training data.
        val_files: List of file paths for validation data.
        n_epochs: Number of epochs to train each model. Defaults to 3.
        device: Device to train on ('cuda' or 'cpu'). Defaults to 'cuda'.
        save_dir: Directory to save experiment results. Defaults to
                 'next_token_prediction_model/study_results'.

    Returns:
        List of ExperimentData objects, one for each hyperparameter combination.

    Side Effects:
        Creates JSON files in save_dir with results from each trial.
        File format: study_{study_num}_trial_{trial_num}.json
    """
    i = 0
    results = []

    for hyperparams in hyperparameters_setup_generator(fixed_config, trial_config):
        print("Trial nr", i)
        print("Hyperparameters:")
        print(asdict(hyperparams))

        experiment_result = create_and_train_model(
            train_files=train_files,
            val_files=val_files,
            hyperparameters=hyperparams,
            trial_params=list(trial_config.keys()),
            n_epochs=n_epochs,
            device=device,
        )
        results.append(experiment_result)

        if save_dir:
            file_name = f"study_{study_num}_trial_{i}.json"
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = save_path / file_name
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(asdict(experiment_result), f, indent=2)
            i += 1

    return results


def get_best_config(
    results: List[ExperimentData],
) -> Tuple[ExperimentData, Dict[str, any]]:
    """
    Find the best performing configuration from grid search results.

    Sorts results by best_score in descending order and extracts the
    configuration of trial parameters from the top result.

    Args:
        results: List of ExperimentData objects from grid search.

    Returns:
        Tuple containing:
            - ExperimentData object with best score
            - Dictionary mapping trial parameter names to their best values

    Example:
        >>> best_result, best_config = get_best_config(results)
        >>> print(f"Best accuracy: {best_result.best_score}")
        >>> print(f"Best learning rate: {best_config['lr']}")
    """
    results.sort(reverse=True)
    best_result = results[0]
    trial_params = best_result.trial_params
    params = best_result.params
    best_config = {}

    for param in trial_params:
        best_config[param] = params.get(param)

    return best_result, best_config


if __name__ == "__main__":

    SEED = 42

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    TOKENIZER = [
        "spm_2k.model",
        "spm_4k.model",
        "spm_8k.model",
        "spm_12k.model",
        "spm_16k.model",
    ]
    EMBEDDING_DIM = [256, 320]

    N_LAYERS = [2, 3, 4]
    HIDDEN_UNITS = [256, 384, 512]

    DROPOUT = [0.2, 0.25, 0.3]
    WEIGHT_DECAY = [1e-2, 1e-3, 1e-4]

    BATCH_SIZE = [32, 64, 128, 256]
    SEQ_LEN = [32, 64, 128, 256]
    LEARNING_RATE = [5e-4, 1e-3, 2e-3]

    study_1 = {
        "fixed_config": {
            "n_layers": 3,
            "hidden_units": 512,
            "dropout": 0.2,
            "weight_decay": 1e-2,
            "batch_size": 128,
            "seq_len": 64,
            "lr": 1e-3,
        },
        "trial_config": {
            "tokenizer": TOKENIZER,
            "emb_dim": EMBEDDING_DIM,
        },
    }

    study_2 = {
        "fixed_config": {
            "dropout": 0.2,
            "weight_decay": 1e-2,
            "batch_size": 128,
            "seq_len": 64,
            "lr": 1e-3,
        },
        "trial_config": {
            "n_layers": N_LAYERS,
            "hidden_units": HIDDEN_UNITS,
        },
    }

    study_3 = {
        "fixed_config": {
            "batch_size": 128,
            "seq_len": 64,
            "lr": 1e-3,
        },
        "trial_config": {
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
        },
    }

    study_4 = {
        "fixed_config": {},
        "trial_config": {
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "lr": LEARNING_RATE,
        },
    }

    configs = [study_1, study_2, study_3, study_4]

    files_paths = glob.glob("./*.txt")
    random.shuffle(files_paths)
    n = int(0.5 * len(files_paths))
    files_paths = files_paths[:n]
    m = max(int(1e-4 * len(files_paths)), 1) * 5
    train_files = files_paths[:-m]
    val_files = files_paths[-m:]

    for study_num, config in enumerate(configs):
        results = run_grid_search(
            study_num=study_num,
            fixed_config=config.get("fixed_config"),
            trial_config=config.get("trial_config"),
            train_files=train_files,
            val_files=val_files,
        )
        best_result, best_config = get_best_config(results)
        print("=" * 50)
        print("BEST CONFIG:")
        print(best_config)
        print("TOP 5 ACCURACY:", best_result.best_score)

        for next_config in configs[study_num + 1 :]:
            for param, val in best_config.items():
                next_config["fixed_config"][param] = val
