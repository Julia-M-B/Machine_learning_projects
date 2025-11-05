import numpy as np
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from models import Base, PositiveSample, TrainingExample
from utils import Vocab, CONFIG_PATH


class TrainingExamplesGenerator:
    def __init__(self, vocab: Vocab, ps_model, examples_model, k: int = 5, table_size: int = 1e8):
        self._vocab = vocab
        self._vocab_size = len(self._vocab.words_counter)
        self._ps_model = ps_model
        self._examples_model = examples_model
        self._k = k
        self._table_size = table_size
        self._unigram_table = self.create_unigram_table()

    def create_unigram_table(self):
        unigram_table = []
        power = 0.75
        norm = sum(count ** power for count in list(self._vocab.words_counter.values())[1:])
        for word, count in list(self._vocab.words_counter.items())[1:]:
            token = self._vocab.get_idx_for_word(word)
            token_appearances = round(count ** power / norm * self._table_size)
            unigram_table.extend([token] * token_appearances)

        return unigram_table

    def draw_negative_tokens(self, center_token, ctx_token):
        negative_tokens = []
        while len(negative_tokens) < self._k:
            neg_t = np.random.randint(1, self._vocab_size)
            if neg_t != center_token and neg_t != ctx_token:
                negative_tokens.append(neg_t)

        return negative_tokens

    def create_final_examples(self, positive_samples):
        examples = []
        for sample in positive_samples:
            center, pos_token = sample
            neg_tokens = self.draw_negative_tokens(center, pos_token)
            example = self._examples_model(target=center, positive=pos_token, negative_tokens=neg_tokens)
            examples.append(example)

        return examples

    def get_positive_samples_data(self, session, batch_size):
        query = session.query(self._ps_model).yield_per(batch_size)
        batch = []
        for ps_object in query:
            batch.append((ps_object.center_token, ps_object.context_token))
            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def update_examples_database(self, session, batch_size):
        samples_gen = self.get_positive_samples_data(session=session,
                                                             batch_size=batch_size)
        total_batches = session.query(PositiveSample).count() // batch_size
        print(total_batches)

        for samples in tqdm(samples_gen, total=total_batches):
            examples = self.create_final_examples(samples)
            try:
                session.add_all(examples)
                session.commit()
            except:
                session.rollback()
                raise



def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    db_path = config["database"]["db-path"]
    db_host = config["database"]["host"]
    batch_size = config["database"]["batch-size"]

    engine = create_engine(f"{db_host}{db_path}", echo=False)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    vocab = Vocab()
    words_counter_name = config["vocabulary"]["words-counter-name"]
    vocab.load_words_counter_from_file(words_counter_name)

    vocab.create_word_to_idx_mapping()
    vocab.create_idx_to_word_mapping()

    examples_gen = TrainingExamplesGenerator(vocab=vocab, ps_model=PositiveSample, examples_model=TrainingExample)

    session.query(TrainingExample).delete()
    session.commit()

    examples_gen.update_examples_database(session=session, batch_size=batch_size)

    session.close()


if __name__ == "__main__":
    main()