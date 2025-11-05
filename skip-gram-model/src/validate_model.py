import torch
import yaml
import os
from utils import CONFIG_PATH, Vocab
from skip_gram_model import SkipGramModel, load_checkpoint, \
    find_most_similar_words
import torch.nn.functional as F
import torch.optim as optim

def main():
    # Load configuration
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Model configuration
    learning_rate = config["model"]["training"].get("learning-rate", 0.001)
    embedding_dim = config["model"].get("embedding-dim", 300)

    # Checkpoint directory
    checkpoint_dir = config["model"]["training"].get("checkpoint-dir",
                                                     "../training_checkpoints")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load vocabulary to get vocab size
    vocab = Vocab()
    words_counter_path = config["vocabulary"].get(
        "lemmatized-words-counter-path", "../lemmatized_words_counter.pkl")
    vocab.load_words_counter_from_file(words_counter_path)
    vocab.create_word_to_idx_mapping()
    vocab.create_idx_to_word_mapping()

    vocab_size = len(vocab.word2idx)
    print(f"Vocabulary size: {vocab_size}")


    # Initialize model
    model = SkipGramModel(vocab_size=vocab_size, embedding_dim=embedding_dim)
    model = model.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load the best model for evaluation
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_checkpoint_path):
        print(
            f"\nLoading best model from {best_checkpoint_path} for evaluation...")
        load_checkpoint(model, optimizer, best_checkpoint_path, device)
        model.eval()

    print("\n" + "=" * 50)
    print("Evaluating best model:")
    print("=" * 50)

    s = "król"
    words = find_most_similar_words(model, vocab=vocab, word=s, top_k=5)
    print(f"\nMost similar words to '{s}':")
    for word, score in words:
        print(f"  {word}: {score:.4f}")

    s2 = "życie"
    words2 = find_most_similar_words(model, vocab=vocab, word=s2, top_k=5)
    print(f"\nMost similar words to '{s2}':")
    for word, score in words2:
        print(f"  {word}: {score:.4f}")

    ###############################################
    ######### WORDS ARITHMETICS ###################
    # `king - man + woman` should equal to `queen`

    king_idx = vocab.get_idx_for_word("król")
    man_idx = vocab.get_idx_for_word("mężczyzna")
    if man_idx == 0 and "mężczyzna" not in vocab.word2idx:
        raise ValueError(f"Word 'mężczyzna' not found in vocabulary")

    woman_idx = vocab.get_idx_for_word("kobieta")
    if woman_idx == 0 and "kobieta" not in vocab.word2idx:
        raise ValueError(f"Word 'kobieta' not found in vocabulary")

    model.eval()
    with torch.no_grad():
        king_emb = model.embeddings(
            torch.LongTensor([king_idx]).to(device))
        man_emb = model.embeddings(
            torch.LongTensor([man_idx]).to(device))
        woman_emb = model.embeddings(
            torch.LongTensor([woman_idx]).to(device))

        target_emb = king_emb - man_emb + woman_emb

        target_emb = F.normalize(target_emb, p=2, dim=1)

        all_embeddings = model.embeddings.weight
        all_embeddings = F.normalize(all_embeddings, p=2, dim=1)

        similarities = torch.mm(target_emb, all_embeddings.t()).squeeze(0)

        top_k_values, top_k_indices = torch.topk(similarities, 5)

        results = []
        for idx, score in zip(top_k_indices.cpu().numpy(),
                              top_k_values.cpu().numpy()):
            similar_word = vocab.get_word_for_idx(int(idx))
            results.append((similar_word, float(score)))

        print("\n" + "#" * 50)
        print("`King - man + woman` is most similar to:")
        for word, score in results:
            print(f"  {word}: {score:.4f}")


if __name__ == "__main__":
    main()
