from tqdm import tqdm
from utils import clean, get_files_paths, CONFIG_PATH
import yaml
from pathlib import Path


def lemmatize_text_file(lemmatized_base_path: str, file_path: str, lemmatizer):
    path = Path(file_path)
    with open(file_path) as f:
        data = clean(f.read())  # wstępny preprocessing tekstu
        doc = lemmatizer(data)
        data = " ".join([token.lemma_.lower() for token in doc])
    with open(f"{lemmatized_base_path}/{path.parent.name}/{path.name}", "w") as f_out:
        f_out.write(data)



def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    database_path = config["data"]["data-dir"]
    datasets_names = config["data"]["datasets"]
    lemmatize = config["lemmatization"]["lemmatize"]
    lemmatizer = config["lemmatization"]["lemmatizer"] if \
    config["lemmatization"]["lemmatizer"] else "pl_core_news_lg"

    if lemmatize:
        import spacy
        lemmatizer = spacy.load(lemmatizer)

    all_files = len(list(get_files_paths(database_path, datasets_names)))
    files_paths = get_files_paths(database_path, datasets_names)

    for file_path in tqdm(files_paths,
                          total=all_files):  # tqdm pozwala na monitoring progresu
        lemmatize_text_file(lemmatized_base_path="../../lemmatized_data",
                            file_path=file_path,
                            lemmatizer=lemmatizer)


if __name__ == "__main__":
    main()
