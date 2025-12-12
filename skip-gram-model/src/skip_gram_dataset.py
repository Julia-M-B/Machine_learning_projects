import torch
from torch.utils.data import IterableDataset
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from models import TrainingExample
import numpy as np
import yaml
from utils import CONFIG_PATH


class SkipGramIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset that reads training examples from database (PostgreSQL or SQLite) in batches.
    This allows handling large datasets that don't fit in RAM and supports concurrent connections.
    """
    
    def __init__(self, db_path: str, db_host: str, batch_size: int = 512, 
                 shuffle: bool = False, validation_split: float = 0.1, 
                 is_validation: bool = False):
        """
        Args:
            db_path: Database connection string (PostgreSQL) or path (SQLite)
                     For PostgreSQL: "postgresql://user:password@host:port/database"
                     For SQLite: path to database file
            db_host: Database host prefix (e.g., "sqlite:///" for SQLite, empty for PostgreSQL)
            batch_size: Number of examples to return per iteration
            shuffle: Whether to shuffle the data (note: shuffling in IterableDataset 
                    is limited - it shuffles within each batch)
            validation_split: Fraction of data to use for validation
            is_validation: If True, return validation set; if False, return training set
        """
        self.db_path = db_path
        self.db_host = db_host
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.is_validation = is_validation

        self.train_records_id, self.val_records_id = self._get_records_id()
        

    def _get_session(self):    
        # Create database connection
        # For PostgreSQL, db_path contains the full connection string
        # For SQLite, it would be db_host + db_path
        connection_string = self.db_path if self.db_path.startswith("postgresql://") else f"{self.db_host}{self.db_path}"
        engine = create_engine(connection_string, echo=False)
        Session = sessionmaker(bind=engine)
        return Session()

    def _get_records_id(self):
        """
        Creates array of records ids (shuffled if needed) and then splits it
        to training records and validation records.
        """
        session = self._get_session()
        n_records = session.query(TrainingExample).count()  # get number of training examples
        session.close()

        val_idx = int((1 - self.validation_split) * n_records)  # split index for training and validation datasets

        records_id = np.arange(1, n_records, step=1)  # create records id array

        if self.shuffle:
            np.random.shuffle(records_id)

        return records_id[:val_idx].tolist(), records_id[val_idx:].tolist()
        
    def __iter__(self):
        """
        Iterator that yields batches of training examples as PyTorch tensors.
        """
        session = self._get_session()

        try:
            # Select the appropriate records ID array
            if self.is_validation:
                records_id_array = self.val_records_id
            else:
                records_id_array = self.train_records_id
            
            # Calculate number of batches
            n_batches = len(records_id_array) // self.batch_size + 1
            
            # Iterate through all batches
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(records_id_array))
                
                # Get the IDs for this batch
                batch_ids = records_id_array[start_idx:end_idx]

                if batch_ids:
                    # Query database for records with these IDs
                    records = session.query(TrainingExample).filter(TrainingExample.id.in_(batch_ids)).all()

                    # Extract data from records
                    batch = []
                    for record in records:
                        target = record.target
                        positive = record.positive
                        negatives = record.negative_tokens

                        batch.append((target, positive, negatives))

                    # Yield batch if it has any valid examples
                    yield self._convert_to_tensors(batch)
                
        finally:
            session.close()

    @staticmethod
    def _convert_to_tensors(batch):
        """
        Convert a batch of examples to PyTorch tensors.
        
        Args:
            batch: List of tuples (target, positive, negatives)
            
        Returns:
            Dictionary with 'target', 'positive', and 'negatives' tensors
        """
        targets = []
        positives = []
        negatives_list = []
        
        for target, positive, negatives in batch:
            targets.append(target)
            positives.append(positive)
            negatives_list.append(list(negatives))
        
        # Convert to tensors
        target_tensor = torch.LongTensor(targets)
        positive_tensor = torch.LongTensor(positives)
        negatives_tensor = torch.LongTensor(negatives_list)  # Shape: [batch_size, num_negatives]

        return {
            'target': target_tensor,
            'positive': positive_tensor,
            'negatives': negatives_tensor
        }

def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Database configuration
    db_path = config["database"]["db-path"]
    db_host = config["database"]["host"]

    dataset = SkipGramIterableDataset(
        db_path=db_path,
        db_host=db_host,
        batch_size=512,
        shuffle=True,
        is_validation=False
    )

    batch_count = 0
    for batch in iter(dataset):
        batch_count += 1
        print(f"Batch {batch_count}: {batch['target'].shape[0]} examples")
        print(batch)
        if batch_count >= 1:  # Just show first 5 batches
            break


if __name__ == "__main__":
    main()
