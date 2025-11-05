import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import os

from skip_gram_dataset import SkipGramIterableDataset
from utils import CONFIG_PATH, Vocab


class SkipGramModel(nn.Module):
    """
    Skip-gram model with negative sampling.
    
    Uses a single shared embedding layer for both target words and context words.
    This is simpler and more parameter-efficient than using separate input/output embeddings.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 300):
        """
        Args:
            vocab_size: Size of vocabulary (number of unique tokens)
            embedding_dim: Dimension of word embeddings (default: 300)
        """
        super(SkipGramModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # Single shared embedding layer for both target and context words

        self._init_embeddings()  # Initialize embeddings with uniform distribution
    
    def _init_embeddings(self):
        """
        Initialize embeddings with uniform distribution in range [-0.5/embedding_dim, 0.5/embedding_dim]
        """
        torch.nn.init.xavier_uniform_(self.embeddings.weight)
    
    def forward(self, target, positive, negatives):
        """
        Forward pass of the skip-gram model with negative sampling.
        
        Args:
            target: LongTensor of shape [batch_size] - target/center word indices
            positive: LongTensor of shape [batch_size] - positive context word indices
            negatives: LongTensor of shape [batch_size, num_negatives] - negative context word indices
            
        Returns:
            loss: Scalar tensor representing the negative sampling loss
        """
        # Get embeddings from single shared embedding layer
        # Shape: [batch_size, embedding_dim]
        target_emb = self.embeddings(target)
        
        # Shape: [batch_size, embedding_dim]
        positive_emb = self.embeddings(positive)
        
        # Shape: [batch_size, num_negatives, embedding_dim]
        negatives_emb = self.embeddings(negatives)
        
        # Compute positive score: dot product between target and positive embeddings
        # Shape: [batch_size]
        positive_score = torch.sum(target_emb * positive_emb, dim=1)
        
        # Compute negative scores: dot products between target and negative embeddings
        # Shape: [batch_size, num_negatives]
        negative_scores = torch.bmm(
            target_emb.unsqueeze(1),  # [batch_size, 1, embedding_dim]
            negatives_emb.transpose(1, 2)  # [batch_size, embedding_dim, num_negatives]
        ).squeeze(1)  # [batch_size, num_negatives]
        
        # Negative sampling loss
        # We want to maximize positive_score and minimize negative_scores
        # Loss = -log(sigmoid(positive_score)) - sum(log(sigmoid(-negative_scores)))
        positive_loss = -torch.log(torch.sigmoid(positive_score) + 1e-10)
        negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_scores) + 1e-10), dim=1)
        
        loss = torch.mean(positive_loss + negative_loss)
        
        return loss
    
    def get_embedding(self, word_idx):
        """
        Get embedding for a word index.
        
        Args:
            word_idx: Integer index of the word
            
        Returns:
            Embedding vector as a numpy array
        """
        with torch.no_grad():
            return self.embeddings(torch.LongTensor([word_idx])).numpy()[0]

def train_epoch(model, dataloader, optimizer, device, epoch):
    """
    Train the model for one epoch.

    Returns:
        Average loss for the epoch
    """
    model.train()  # set model in the training mode
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        # Move tensors to device
        target = batch['target'].to(device)
        positive = batch['positive'].to(device)
        negatives = batch['negatives'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        loss = model(target, positive, negatives)

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, device):
    """
    Validate the model.

    Returns:
        Average loss on validation set
    """
    model.eval()  # set model in the evaluation mode
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move tensors to device
            target = batch['target'].to(device)
            positive = batch['positive'].to(device)
            negatives = batch['negatives'].to(device)

            # Forward pass
            loss = model(target, positive, negatives)

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, best=False):
    """
    Save model checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    if best:
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Load model checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def find_most_similar_words(model, vocab, word: str, top_k: int = 5):
    """
    Find the most similar words to a given word using cosine similarity.

    Args:
        model: Trained SkipGramModel instance
        vocab: Vocab object with word2idx and idx2word mappings
        word: The word to find similar words for (string)
        top_k: Number of most similar words to return (default: 5)

    Returns:
        List of tuples (similar_word, similarity_score) sorted by similarity (highest first)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get word index
    word = word.lower()
    word_idx = vocab.get_idx_for_word(word)

    # Check if word is in vocabulary
    if word_idx == 0 and word not in vocab.word2idx:
        raise ValueError(f"Word '{word}' not found in vocabulary")

    model.eval()
    with torch.no_grad():
        # Get embedding for the target word
        # Shape: [1, embedding_dim]
        target_emb = model.embeddings(torch.LongTensor([word_idx]).to(device))

        # Normalize target embedding for cosine similarity
        target_emb = F.normalize(target_emb, p=2, dim=1)

        # Get all embeddings and normalize
        # Shape: [vocab_size, embedding_dim]
        all_embeddings = model.embeddings.weight
        all_embeddings = F.normalize(all_embeddings, p=2, dim=1)

        # Compute cosine similarity
        # Shape: [vocab_size]
        similarities = torch.mm(target_emb, all_embeddings.t()).squeeze(0)

        # Exclude the target word itself
        similarities[word_idx] = -1.0

        # Get top_k most similar words
        top_k_values, top_k_indices = torch.topk(similarities, top_k)

        # Convert to list of (word, similarity) tuples
        results = []
        for idx, score in zip(top_k_indices.cpu().numpy(),
                              top_k_values.cpu().numpy()):
            similar_word = vocab.get_word_for_idx(int(idx))
            results.append((similar_word, float(score)))

        return results


def main():
    # Load configuration
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    # Database configuration
    db_path = config["database"]["db-path"]
    db_host = config["database"]["host"]
    
    # Model configuration
    batch_size = config["model"]["training"].get("batch-size", 512)
    learning_rate = config["model"]["training"].get("learning-rate", 0.001)
    embedding_dim = config["model"].get("embedding-dim", 300)
    num_epochs = config["model"]["training"].get("num-epochs", 10)
    validation_split = config["model"]["training"].get("validation-split", 0.1)
    
    # Checkpoint directory
    checkpoint_dir = config["model"]["training"].get("checkpoint-dir", "../training_checkpoints")
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load vocabulary to get vocab size
    vocab = Vocab()
    words_counter_path = config["vocabulary"].get("lemmatized-words-counter-path", "../lemmatized_words_counter.pkl")
    vocab.load_words_counter_from_file(words_counter_path)
    vocab.create_word_to_idx_mapping()
    vocab.create_idx_to_word_mapping()
    
    vocab_size = len(vocab.word2idx)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create datasets
    train_dataset = SkipGramIterableDataset(
        db_path=db_path,
        db_host=db_host,
        batch_size=batch_size,
        shuffle=True,
        validation_split=validation_split,
        is_validation=False
    )
    
    val_dataset = SkipGramIterableDataset(
        db_path=db_path,
        db_host=db_host,
        batch_size=batch_size,
        shuffle=False,
        validation_split=validation_split,
        is_validation=True
    )
    
    # Create dataloaders
    # Note: For IterableDataset, num_workers should be 0 or 1 to avoid database connection issues
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)
    
    # Initialize model
    model = SkipGramModel(vocab_size=vocab_size, embedding_dim=embedding_dim)
    model = model.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler to reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )

    # Early stopping parameters
    early_stopping_patience = config["model"]["training"].get("early-stopping-patience", 3)
    early_stopping_counter = 0

    # Training loop
    best_val_loss = float('inf')
    start_epoch = 0

    # Load checkpoint if exists
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path, device)
        start_epoch += 1

    print(f"Starting training from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch + 1)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss = validate(model, val_loader, device)
        print(f"Validation Loss: {val_loss:.4f}")

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch + 1, val_loss, checkpoint_dir, best=False)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0  # Reset counter on improvement
            save_checkpoint(model, optimizer, epoch + 1, val_loss, checkpoint_dir, best=True)
            print(f"New best model saved! Validation loss: {val_loss:.4f}")
        else:
            early_stopping_counter += 1
            print(f"No improvement. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()

