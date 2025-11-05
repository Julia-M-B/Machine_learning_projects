import yaml

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from models import PositiveSample, Base
from collections import deque

from utils import clean, get_single_words, Vocab, SubsamplingHelper, \
    CONFIG_PATH, get_files_paths


class SlidingWindow(deque):
    def __init__(self, context_size: int = 5):
        self._context_size = context_size
        super().__init__(maxlen=context_size * 2 + 1)

    def is_ready(self) -> bool:
        """
        funkcja zwraca informację, czy kolejka już jest zapełniona,
        czyli czy można dla danego słowa centralnego utworzyć pełny kontekst
        """
        return len(list(self)) == self.maxlen

    def get_context_words(self):
        """
        funkcja zwraca listę słów tworzących kontekst dla pewnego słowa centralnego
        (środkowe słowo kolejki) - słowa kontekstowe zwracane są w kolejności
        pierwsze od lewej, pierwsze od prawej, drugie od lewej, drugie od prawej, ...
        """
        if self.is_ready():
            left_context = list(self)[:self._context_size][::-1]
            right_context = list(self)[self._context_size + 1:]

            context = []
            for i in range(self._context_size):
                context.append(left_context[i])
                context.append(right_context[i])
            return context
        else:
            print("Kolejka się nie zapełniła.")

    def get_center_word(self):
        """
        zwraca środkowe słowo kolejki
        """
        if self.is_ready():
            return list(self)[self._context_size]
        else:
            print("Kolejka się nie zapełniła.")

    def get_center_context_pair(self):
        if self.is_ready():
            return self.get_center_word(), self.get_context_words()
        else:
            print("Kolejka się nie zapełniła.")

class PositiveSamples:
    """
    Klasa, która zarządza tworzeniem "pozytywnych" przykładów
    i zapisaniem ich do bazy w formie tokenów.

    Dostając ścieżkę do pliku, musi go wczytać i preprocesować.
    Następnie przechodzi po tekstach swoim oknem.
    Stosuje subsampling do słowa centralnego: jeśli słowo centralne
    jest odrzucone, to okno idzie dalej, jeśli słowo zostaje, to
    do każdego słowa z kontekstu stosuje subsampling.
    Ze słów, które zostaną, tworzy pary: słowo centralne, słowo kontekstowe.
    Przypisuje pary do listy par.
    Gdy lista się zapełni do odpowiedniej liczby, baza danych jest updatowana.
    """
    def __init__(self, vocab: Vocab, subsampling_helper: SubsamplingHelper, db_model):
        self._vocab = vocab
        self._subsampling_helper = subsampling_helper
        self._db_model = db_model


    def positive_samples_generator(self, data, batch_size):
        sliding_window  = SlidingWindow()
        positive_samples = []
        counter = 0
        for word in get_single_words(data):
            sliding_window.append(word)
            if sliding_window.is_ready():
                center_word = sliding_window.get_center_word()
                if not self._subsampling_helper.subsample_word(center_word):
                    continue
                center_token = self._vocab.get_idx_for_word(center_word)
                context_words = sliding_window.get_context_words()
                context_tokens = [self._vocab.get_idx_for_word(word) for word in context_words]
                for context_token in context_tokens:
                    positive_samples.append((center_token, context_token))
                    counter += 1

            if counter == batch_size:
                yield positive_samples
                counter = 0
                positive_samples = []

        yield positive_samples

    def update_positive_samples_table(self, data, batch_size, session):
        for samples in self.positive_samples_generator(data, batch_size):
            try:
                positive_samples = []
                for center, ctx in samples:
                    ps = self._db_model(center_token=center, context_token=ctx)
                    positive_samples.append(ps)

                session.add_all(positive_samples)
                session.commit()
            except:
                session.rollback()
                raise

def proces_file_and_update_database(file_path, ps_maker: PositiveSamples, batch_size: int, session):
    with open(file_path) as f:
        data = clean(f.read()) # wstępny preprocessing tekstu
        ps_maker.update_positive_samples_table(data, batch_size, session)


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    datasets_names = config["data"]["datasets"]
    db_path = config["database"]["db-path"]
    db_host = config["database"]["host"]

    lemmatize = config["lemmatization"]["lemmatize"]

    if lemmatize:
        database_path = config["data"]["lemmatized-data-dir"]
    else:
        database_path = config["data"]["data-dir"]

    engine = create_engine(f"{db_host}{db_path}", echo=False )
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    session.query(PositiveSample).delete()
    session.commit()

    vocab = Vocab()
    words_counter_name = config["vocabulary"]["words-counter-name"]
    vocab.load_words_counter_from_file(words_counter_name)

    vocab.create_word_to_idx_mapping()
    vocab.create_idx_to_word_mapping()

    subsampling_helper = SubsamplingHelper(vocab=vocab, p=95)

    ps_maker = PositiveSamples(vocab=vocab, subsampling_helper=subsampling_helper, db_model=PositiveSample)

    all_files = len(list(get_files_paths(database_path, datasets_names)))
    files_paths = get_files_paths(database_path, datasets_names)

    for file_path in tqdm(files_paths, total=all_files):  # tqdm pozwala na monitoring progresu
        proces_file_and_update_database(file_path, ps_maker, batch_size=10_000, session=session)
        # break

    session.close()



if __name__ == "__main__":
    main()
