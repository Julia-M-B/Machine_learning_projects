import requests

URL = "http://localhost:8000"

# Test połączenia
response = requests.get(f"{URL}/")
print("Status API:", response.json())


# Generowanie tekstu
def generate_text(prompt):
    response = requests.post(
        f"{URL}/conversation",
        json={
            "history": [],
            "prompt": prompt,
        }
    )

    response.raise_for_status()
    return response.json()


# Przykład użycia
result = generate_text("Ala ma kota i kot ma Alę")
print("\nWygenerowany tekst:")
print(result["generated_text"])
print("\nHistoria:")
print(result["history"])