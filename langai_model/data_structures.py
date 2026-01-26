from pydantic import BaseModel
from typing import List, Tuple


class UserUtterance(BaseModel):
    """
    Docelowo klasa dla pojedynczej wypowiedzi użytkownika.
    Przechowuje tekst samej wypowiedzi (czyli audio już zmienione na tekst).
    """
    history: List[Tuple[str, str]]  # lista tupli wypowiedzi <user, model>
    prompt: str  # najnowsza wypowiedź usera

class ConversationModelResponse(BaseModel):
    history: List[Tuple[str, str]]  # lista tupli wypowiedzi <user, model>
    prompt: str  # najnowsza wypowiedż usera
    generated_text: str  # odpowiedź modelu na najnowszą wypowiedź usera
