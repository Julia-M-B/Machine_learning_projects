import torch
from fastapi import FastAPI, HTTPException, Header
import os
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_structures import UserUtterance, ConversationModelResponse
from conversations_utils import get_dummy_response
from typing import Optional


app = FastAPI()

MODEL_NAME = ""  # name of the model from hugging face
API_KEY = os.environ.get("HUGGING_FACE_API_KEY")

device = "cuda" if torch.cuda.is_available() else "cpu"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "Local LLM API",
        "model": MODEL_NAME,
        "device": device
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}


@app.post("/conversation", response_model=ConversationModelResponse)
async def generate(
        request: UserUtterance,
        x_api_key: Optional[str] = Header(None)
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # try:
    #     inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
    #
    #     # Generowanie
    #     with torch.no_grad():
    #         outputs = model.generate(
    #             **inputs,
    #             max_length=request.max_length,
    #             temperature=request.temperature,
    #             top_p=request.top_p,
    #             do_sample=True,
    #             pad_token_id=tokenizer.eos_token_id
    #         )
    #
    #     # Dekodowanie
    #     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # dummy responses
    try:
        generated_text = get_dummy_response(request.prompt)
        history = request.history
        history.append((request.prompt, generated_text))

        return ConversationModelResponse(
            history=history,
            prompt=request.prompt,
            generated_text=generated_text
        )

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Generation error: {str(e)}")


if __name__ == "__main__":
    # host="0.0.0.0" umożliwia dostęp z innych komputerów
    # host="127.0.0.1" ogranicza tylko do lokalnego komputera
    uvicorn.run(app, host="0.0.0.0", port=8000)