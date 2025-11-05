# Polish Skip-Gram Model with Negative Sampling

A PyTorch implementation of a Skip-Gram model with negative sampling for Polish language word embeddings. This project provides a complete pipeline for preprocessing Polish text data, building vocabulary, and training word embeddings using the skip-gram architecture.

## 📋 Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Data Preprocessing](#1-data-preprocessing)
  - [Vocabulary Creation](#2-vocabulary-creation)
  - [Training the Model](#3-training-the-model)
  - [Model Validation](#4-model-validation)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Author](#author)

## Overview

This project implements a Skip-Gram model with negative sampling for learning Polish word embeddings. The model learns distributed representations of words by predicting context words from a target word. The implementation includes:

- **Text preprocessing**: Cleaning, normalization, and optional lemmatization
- **Vocabulary management**: Word frequency counting and index mapping
- **Database integration**: SQLite database for efficient handling of large datasets
- **Model training**: PyTorch-based training with checkpointing and early stopping
- **Model evaluation**: Word similarity search and word arithmetic

## Requirements

### Python Dependencies

```bash
pip install -r requirements.txt
```

### Polish Language Model for spaCy

If using lemmatization, download the Polish language model:

```bash
python -m spacy download pl_core_news_lg
```

## Project Structure

```
skip-gram-model/
├── config.yaml                 # Configuration file
├── README.md                   # This file
├── src/                        # Source code
│   ├── __init__.py
│   ├── create_vocabulary.py   # Vocabulary creation script
│   ├── lemmatize_texts.py      # Text lemmatization script
│   ├── models.py               # SQLAlchemy database models
│   ├── negative_sampling_dataset.py
│   ├── positive_samples_dataset.py
│   ├── skip_gram_dataset.py   # PyTorch dataset implementation
│   ├── skip_gram_model.py     # Main model and training code
│   ├── utils.py               # Utility functions and Vocab class
│   └── validate_model.py      # Model validation script
├── training_checkpoints/       # Saved model checkpoints
├── words_counter.pkl           # Word frequency counter (raw)
└── lemmatized_words_counter.pkl # Word frequency counter (lemmatized)
```

## Configuration

The project is configured via `config.yaml`. Key configuration options:

### Data Settings
```yaml
data:
  data-dir: "../../data"              # Raw data directory
  lemmatized-data-dir: "../../lemmatized_data"  # Lemmatized data directory
  datasets:                           # List of datasets to use
    - "wolne_lektury_corpus"
```

### Database Settings
```yaml
database:
  db-path: "sqlite.db"                # SQLite database path
  host: "sqlite:///"                  # Database host
  batch-size: 1024                    # Batch size for database operations
```

### Model Settings
```yaml
model:
  embedding-dim: 300                  # Embedding dimension
  training:
    batch-size: 512                   # Training batch size
    learning-rate: 0.001               # Initial learning rate
    num-epochs: 10                    # Number of training epochs
    validation-split: 0.1             # Validation set fraction
    checkpoint-dir: "../training_checkpoints"
    early-stopping-patience: 3        # Early stopping patience
```

### Vocabulary Settings
```yaml
vocabulary:
  context-size: 5                     # Context window size
  vocabulary-size: 10000              # Maximum vocabulary size
  batch-size: 1024
  words-counter-path: "../words_counter.pkl"
  lemmatized-words-counter-path: "../lemmatized_words_counter.pkl"
```

### Lemmatization Settings
```yaml
lemmatization:
  lemmatize: true                     # Enable/disable lemmatization
  lemmatizer: "pl_core_news_lg"       # spaCy model name
```

## Usage

### 1. Data Preprocessing

#### Step 1.1: Text Lemmatization (Optional)

If you want to use lemmatized text, run the lemmatization script:

```bash
cd src
python lemmatize_texts.py
```

This script:
- Reads text files from the configured data directory
- Cleans and normalizes the text
- Applies lemmatization using spaCy's Polish model
- Saves lemmatized text to the output directory

#### Step 1.2: Create Vocabulary

Create vocabulary from your text files:

```bash
cd src
python create_vocabulary.py
```

This script:
- Processes all text files in the configured datasets
- Counts word frequencies
- Saves the word counter to a `.pkl` file
- Creates word-to-index and index-to-word mappings

**Note**: Make sure to configure `words-counter-name` in `config.yaml` before running.

### 2. Prepare Training Data

Before training, you need to:
1. Create positive samples (center-context pairs) from your text
2. Generate negative samples for negative sampling
3. Store training examples in the SQLite database

These steps are typically handled by separate scripts that should be run before training.

### 3. Training the Model

Train the skip-gram model:

```bash
cd src
python skip_gram_model.py
```

The training process:
- Loads vocabulary from the configured counter file
- Creates train/validation datasets from the SQLite database
- Initializes the Skip-Gram model with the specified embedding dimension
- Trains using Adam optimizer with learning rate scheduling
- Saves checkpoints after each epoch
- Saves the best model based on validation loss
- Implements early stopping if validation loss doesn't improve

**Training Output**:
- `checkpoint_epoch_{N}.pt`: Checkpoint for each epoch
- `best_model.pt`: Best model based on validation loss

### 4. Model Validation

Evaluate the trained model:

```bash
cd src
python validate_model.py
```

This script:
- Loads the best model checkpoint
- Finds most similar words to example queries
- Performs word arithmetic (e.g., "król - mężczyzna + kobieta")

**Example Output**:
```
Most similar words to 'król':
  królowa: 0.8234
  monarcha: 0.7891
  ...

`King - man + woman` is most similar to:
  królowa: 0.7654
  ...
```

## Model Architecture

### Skip-Gram with Negative Sampling

The model implements the Skip-Gram architecture with negative sampling:

1. **Single Shared Embedding Layer**: Uses one embedding layer for both target and context words (parameter-efficient)

2. **Negative Sampling Loss**: 
   - Maximizes similarity between target words and their positive context words
   - Minimizes similarity between target words and randomly sampled negative words
   - Uses 5 negative samples per positive example

3. **Training Process**:
   - Forward pass computes dot products between embeddings
   - Loss combines positive and negative log-likelihoods
   - Backward pass updates embeddings using gradient descent
   - Gradient clipping prevents exploding gradients

### Key Components

- **Embedding Dimension**: 300 (configurable)
- **Context Window**: 5 words on each side (configurable)
- **Negative Samples**: 5 per positive example
- **Optimizer**: Adam with learning rate scheduling
- **Regularization**: Gradient clipping (max_norm=1.0)

## Results

After training, you can:

1. **Find Similar Words**: Query the model for words similar to a given word
2. **Word Arithmetic**: Perform analogical reasoning (e.g., "król - mężczyzna + kobieta ≈ królowa")
3. **Extract Embeddings**: Use learned embeddings for downstream NLP tasks

The model learns semantic and syntactic relationships between Polish words, capturing:
- Semantic similarity (synonyms, related concepts)
- Syntactic relationships (grammatical patterns)
- Word analogies (semantic relationships)

## Author

**Julia Bochniarz-Paziewska**

---

## Additional Notes

- The model uses SQLite for efficient data management, allowing training on datasets larger than available RAM
- Text preprocessing includes: lowercasing, removal of non-letter characters, whitespace normalization
- The vocabulary includes an `<unk>` token (index 0) for out-of-vocabulary words
- Training checkpoints can be used to resume training from a specific epoch
- The model automatically uses CUDA if available, otherwise falls back to CPU

