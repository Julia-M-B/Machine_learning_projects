from src.beam_search import WordPredictionBeamSearch
from src.model import StreamingTextDataset, LSTMLanguageModel, train_epoch, compute_val_ppl
import optuna
from torch.utils.data import DataLoader
import torch
from collections import deque
from typing import List
import sentencepiece as spm
import json
import glob

def get_top_k_accuracy(searcher: WordPredictionBeamSearch, context: str, true_word: str, k: int = 5) -> float:
    top_words = searcher.get_top_k_words(context_text=context, k=k)
    predictions = [word for word, _, _ in top_words]
    return 1.0 if true_word in predictions else 0.0

def evaluate_top_k_words(searcher: WordPredictionBeamSearch, files_paths: List[str], seq_len: int, k: int = 5) -> float:
    preds_counter = 0
    acc5 = 0
    for file in files_paths:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        words = text.split()
        window = deque(maxlen=seq_len+1)
        for word in words:
            window.append(word)
            if len(window) == window.maxlen:
                context = " ".join(window[:-1]) + " "
                target = window[-1]
                acc5 += get_top_k_accuracy(searcher, context, target, k)
                preds_counter += 1

    return acc5 / preds_counter if preds_counter > 0 else 0

def make_objective(fixed_config: dict, trial_config: dict, files_paths: List[str], n_epochs: int, device: str):
    def objective(trial):
        def get_param_value(param_name: str):
            fixed_val = fixed_config.get(param_name)
            trial_val = trial_config.get(param_name)
            return fixed_val if fixed_val else trial.suggest_categorical(param_name, trial_val)

        tokenizer = get_param_value("tokenizer")
        emb_dim = get_param_value("emb_dim")
        n_layers = get_param_value("n_layers")
        hidden_units = get_param_value("hidden_units")
        dropout = get_param_value("dropout")
        weight_decay = get_param_value("weight_decay")
        batch_size = get_param_value("batch_size")
        seq_len = get_param_value("seq_len")
        lr = get_param_value("lr")

        train_files = files_paths[: int(0.95 * len(files_paths))]
        val_files = files_paths[int(0.95 * len(files_paths)):]

        train_ds = StreamingTextDataset(files=train_files, sp_model_path=tokenizer, seq_len=seq_len)
        val_ds = StreamingTextDataset(files=val_files, sp_model_path=tokenizer, seq_len=seq_len)

        train_dl = DataLoader(train_ds, batch_size=batch_size)
        val_dl = DataLoader(val_ds, batch_size=batch_size)

        sp_proc = spm.SentencePieceProcessor()
        sp_proc.load(tokenizer)

        model = LSTMLanguageModel(
            vocab_size=sp_proc.get_piece_size(),
            emb_dim=emb_dim,
            hidden_dim=hidden_units,
            n_layers=n_layers,
            dropout=dropout
        ).to(device)

        searcher = WordPredictionBeamSearch(
            model=model,
            tokenizer=sp_proc
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        save_path = f"{tokenizer}_emb{emb_dim}_nlay{n_layers}_hu{hidden_units}_drop{dropout}_wd{weight_decay}_batch{batch_size}_seq{seq_len}_lr{lr}.json"
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_ppl": [],
            "val_acc5": [],
            "score": [],
        }

        for epoch in range(1, n_epochs + 1):
            print(f"Training epoch {epoch}:")
            train_loss = train_epoch(model, train_dl, optimizer, device)
            print(f"Epoch {epoch} train loss: {train_loss:.4f}")

            val_loss, val_ppl = compute_val_ppl(model, val_dl, device)
            val_acc5 = evaluate_top_k_words(searcher, files_paths=val_files, seq_len=seq_len)
            score = val_ppl + (1 - val_acc5) * 100

            print(f"Epoch {epoch} val loss: {val_loss:.4f} ppl: {val_ppl:.2f}")
            print(f"Epoch {epoch} val acc: {val_acc5:.4f}")
            print(f"Epoch {epoch} score: {score:.4f}")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_ppl"].append(val_ppl)
            history["val_acc5"].append(val_acc5)
            history["score"].append(score)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        return score

    return objective

if __name__ == "__main__":

    DEVICE = "cuda"
    FILES_PATHS = glob.glob("./*.txt")

    TOKENIZER = ["spm_2k.model", "spm_4k.model", "spm_8k.model",
                 "spm_12k.model", "spm_16k.model"]
    EMBEDDING_DIM = [256, 320]

    N_LAYERS = [2, 3]
    HIDDEN_UNITS = [256, 384, 512]

    DROPOUT = [0.2, 0.25, 0.3]
    WEIGHT_DECAY = [1e-2, 1e-3, 1e-4]

    BATCH_SIZE = [32, 64, 128, 256]
    SEQ_LEN = [32, 64, 128, 256]
    LEARNING_RATE = [5e-4, 1e-3, 2e-3]

    study_1 = {
        "fixed_config" : {
            "n_layers" : 3,
            "hidden_units" : 512,
            "dropout" : 0.2,
            "weight_decay" : 1e-2,
            "batch_size" : 128,
            "seq_len" : 64,
            "lr" : 1e-3,
        },
        "trial_config": {
            "tokenizer" : TOKENIZER,
            "emb_dim" : EMBEDDING_DIM,
        },
    }

    study_2 = {
        "fixed_config": {

        },
        "trial_config": {

        },
    }

    study_3 = {
        "fixed_config": {

        },
        "trial_config": {

        },
    }

    study_4 = {
        "fixed_config": {

        },
        "trial_config": {

        },
    }

    configs = [study_1]

    for config in configs:
        objective_fn = make_objective(config.get("fixed_config"), config.get("trial_config"), FILES_PATHS, n_epochs=3, device=DEVICE)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_fn, n_trials=10)

        print("=" * 50)
        print("BEST:")
        print(study.best_params)
        print("PPL:", study.best_value)