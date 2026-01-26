import random

def change_audio_to_text(user_prompt: str) -> str:
    """
    dummy funkcja, która na razie przyjmuje tekst i zwraca tekst.
    Docelowo będzie zamieniała wypowiedź na tekst i pewnie nie bedzie
    to jedna funkcja, tylko bardziej złożony mechanizm.
    :return:
    """
    return user_prompt

# dummy odpowiedź modelu
def get_dummy_response(user_prompt: str) -> str:
    words = user_prompt.split()
    random.shuffle(words)
    return " ".join(words)