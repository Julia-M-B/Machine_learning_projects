import sys

from tqdm import tqdm
from utils import clean, get_files_paths, Vocab, CONFIG_PATH
import yaml



def create_vocabulary_from_file(file_path: str, vocab: Vocab):
    with open(file_path) as f:
        data = clean(f.read()).split(" ")  # wstępny preprocessing tekstu
        vocab.update_words_counter(data)


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    datasets_names = config["data"]["datasets"]

    lemmatize = config["lemmatization"]["lemmatize"]

    if lemmatize:
        database_path = config["data"]["lemmatized-data-dir"]
    else:
        database_path = config["data"]["data-dir"]

    vocab = Vocab()
    words_counter_name = config["vocabulary"]["words-counter-name"]

    all_files = len(list(get_files_paths(database_path, datasets_names)))
    files_paths = get_files_paths(database_path, datasets_names)

    for file_path in tqdm(files_paths,
                          total=all_files):  # tqdm pozwala na monitoring progresu
        create_vocabulary_from_file(file_path, vocab)

    vocab.save_words_counter_to_file(words_counter_name)

    print("Words counter size in bytes:", sys.getsizeof(vocab.words_counter))

    n = 50
    print(f"Words counter first {n} items:")
    for _, item in zip(range(n), vocab.words_counter.items()):
        print(item)


if __name__ == "__main__":
    main()
