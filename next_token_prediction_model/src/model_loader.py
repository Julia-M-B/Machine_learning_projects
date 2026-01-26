"""
Model loader and tokenizer wrapper for LSTM language model.
"""
import os
import torch
import torch.nn as nn
import sentencepiece as spm
from typing import List


class LSTMLanguageModel(nn.Module):
    """LSTM Language Model architecture."""
    
    def __init__(self, vocab_size: int, emb_dim: int = 320, hidden_dim: int = 512,
                 n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, 
                           num_layers=n_layers, batch_first=True, 
                           dropout=dropout if n_layers > 1 else 0.0)
        if self.emb_dim != self.hidden_dim:
            self.projection = nn.Linear(hidden_dim, emb_dim, bias=False)
        self.output = nn.Linear(emb_dim, vocab_size)
        self.output.weight = self.embedding.weight

    def forward(self, input_ids: torch.LongTensor, hidden=None):
        # input_ids: (batch, seq_len)
        emb = self.embedding(input_ids)  # (batch, seq_len, emb_dim)
        out, hidden = self.lstm(emb, hidden)  # out: (batch, seq_len, hidden)
        if self.emb_dim != self.hidden_dim:
            out = self.projection(out)
        logits = self.output(out)  # (batch, seq_len, vocab)
        return logits, hidden


class LSTMModelWrapper:
    """
    Wrapper for LSTM model that provides predict() method for beam search.
    """
    
    def __init__(self, model_path: str, device: str = None, seq_len: int = 64):
        """
        Initialize the model wrapper.
        
        Args:
            model_path: Path to model.pt file
            device: Device to run model on ('cpu' or 'cuda'). If None, auto-detect.
        """
        if device is None:
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            device = 'cpu'
        self.device = torch.device(device)
        
        # Load model state dict
        state_dict = torch.load(model_path, map_location=self.device)

        # Infer model architecture from state dict
        # Get vocab_size from embedding weight
        if 'embedding.weight' in state_dict:
            vocab_size = state_dict['embedding.weight'].shape[0]
        else:
            raise ValueError("Could not infer vocab_size from state dict")

        # Get other hyperparameters from state dict
        emb_dim = state_dict['embedding.weight'].shape[1]
        if 'lstm.weight_ih_l0' in state_dict:
            # LSTM input size is 1/4 of weight_ih_l0 (because it's 4 gates)
            hidden_dim = state_dict['lstm.weight_ih_l0'].shape[0] // 4
        else:
            hidden_dim = 512  # default

        # Count LSTM layers
        n_layers = 0
        while f'lstm.weight_ih_l{n_layers}' in state_dict:
            n_layers += 1
        if n_layers == 0:
            n_layers = 3  # default
        
        # Create model
        self.model = LSTMLanguageModel(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers
        )
        

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        
        self.vocab_size = vocab_size
        self.seq_len = seq_len
    
    def predict(self, context_tokens: List[int]) -> List[float]:
        """
        Predict next token probabilities given context tokens.
        
        Args:
            context_tokens: List of token IDs representing the context
            
        Returns:
            List of probabilities for each token in vocabulary
        """
        if not context_tokens:
            # If no context, return uniform distribution
            return [1.0 / self.vocab_size] * self.vocab_size

        # trim context tokens to proper sequence length
        context_tokens = context_tokens[-self.seq_len:]
        
        # Convert to tensor and add batch dimension
        input_ids = torch.LongTensor([context_tokens]).to(self.device)
        
        with torch.no_grad():
            # Get logits for the last position
            logits, _ = self.model(input_ids)
            # logits shape: (batch=1, seq_len, vocab_size)
            # Get logits for the last token position
            last_logits = logits[0, -1, :]  # (vocab_size,)
            
            # Convert to probabilities using softmax
            probs = torch.softmax(last_logits, dim=0)
            
            # Convert to list
            return probs.cpu().tolist()


class SentencePieceTokenizer:
    """
    Wrapper for SentencePiece tokenizer.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the tokenizer.
        
        Args:
            model_path: Path to .model file (e.g., spm_pl.model)
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.vocab_size = self.sp.get_piece_size()
        self.id2piece = self._create_id_to_piece_mapping()
        self.piece2id = self._create_piece_to_id_mapping()

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        return self.sp.encode(text, out_type=int)
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        return self.sp.decode(token_ids)

    def encode_as_pieces(self, text):
        return self.sp.encode_as_pieces(text)

    def id_to_piece(self, token: int) -> str:
        return self.sp.id_to_piece(token)

    def _create_id_to_piece_mapping(self) -> dict[int, str]:
        id_piece_pairs = [(i, self.id_to_piece(i)) for i in range(self.vocab_size)]
        return dict(id_piece_pairs)

    def _create_piece_to_id_mapping(self) -> dict[str, int]:
        piece_id_pairs = [(self.id_to_piece(i), i) for i in
                          range(self.vocab_size)]
        return dict(piece_id_pairs)

def load_model_and_tokenizer(model_dir: str = None, device: str = None, seq_len: int = 64):
    """
    Convenience function to load both model and tokenizer.
    
    Args:
        model_dir: Directory containing model.pt and spm_pl.model. 
                  If None, uses predictions directory.
        device: Device to run model on. If None, auto-detect.
        
    Returns:
        Tuple of (model_wrapper, tokenizer)
    """
    if model_dir is None:
        # Get directory of this file
        model_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(model_dir, 'model.pt')
    tokenizer_path = os.path.join(model_dir, 'spm_pl.model')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    
    model = LSTMModelWrapper(model_path, device=device, seq_len=seq_len)
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    
    return model, tokenizer

def main():
    model, tokenizer = load_model_and_tokenizer(".", device="cpu")
    text2 = "witaj mam dziś gorszy dzień\nczy wszystko okej\nspoko ale przydałby się tlen\n"
    # print(tokenizer.encode_as_pieces(text2))
    print(tokenizer.encode(text2))

if __name__ == "__main__":
    main()
