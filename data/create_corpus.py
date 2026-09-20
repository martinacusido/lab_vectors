from datasets import load_dataset
import re

print("Carregant el corpus...")

dataset = load_dataset(
    "lucadiliello/bookcorpusopen",
    split="train",
    streaming=True
)

sentences = []

for example in dataset:
    text = example["text"]

    # Convertim salts de línia en espais
    text = text.replace("\n", " ")

    # Separem el text en frases
    parts = re.split(r'(?<=[.!?])\s+', text)

    for sentence in parts:
        sentence = sentence.strip()
        sentence = re.sub(r'\s+', ' ', sentence)

        # Filtres de qualitat
        if not (20 <= len(sentence) <= 300):
            continue

        # Eliminar fragments típics de metadades
        lower = sentence.lower()

        forbidden = [
            "copyright",
            "www.",
            "http",
            "ebook edition",
            "all rights reserved",
            "smashwords",
            "isbn",
            "click here"
        ]

        if any(word in lower for word in forbidden):
            continue

        # Evitar frases amb massa símbols/números
        letters = sum(c.isalpha() for c in sentence)
        if letters < 15:
            continue

        sentences.append(sentence)

        if len(sentences) >= 10000:
            break

    if len(sentences) >= 10000:
        break

print(f"S'han obtingut {len(sentences)} frases.")

with open("data/sentences.txt", "w", encoding="utf-8") as f:
    for sentence in sentences:
        f.write(sentence + "\n")

print("Fitxer creat: data/sentences.txt")
