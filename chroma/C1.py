import csv
import time
import hashlib

import chromadb
import numpy as np
import statistics
from sentence_transformers import SentenceTransformer


OUTPUT_FILE = "results/chroma_embedding_times.csv"


def main():
    print("Carregant el model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connectant amb Chroma...")
    client = chromadb.PersistentClient(path="./chroma_db")
    col_l2 = client.get_collection("sentences_l2")
    col_cos = client.get_collection("sentences_cosine")

    # Recuperem les frases que hem inserit a C0
    data = col_l2.get()
    ids = data["ids"]
    sentences = data["documents"]

    print(f"Frases llegides de Chroma: {len(sentences)}")

    # Chroma no garanteix l'ordre al fer get(), així que ho ordenem numèricament
    sorted_data = sorted(zip(ids, sentences), key=lambda x: int(x[0]))
    ids, sentences = zip(*sorted_data)

    # Convertim a llistes per facilitar les comprovacions
    ids = list(ids)
    sentences = list(sentences)

    # ---------------------------------------------------------
    # COMPROVACIÓ DE LES FRASES
    # ---------------------------------------------------------

    print("\n--- Primeres 5 frases ---")
    for sentence_id, sentence in list(zip(ids, sentences))[:5]:
        print(f"ID {sentence_id}: {sentence}")

    print("\n--- Últimes 5 frases ---")
    for sentence_id, sentence in list(zip(ids, sentences))[-5:]:
        print(f"ID {sentence_id}: {sentence}")

    # ---------------------------------------------------------
    # HASH DEL CORPUS
    # ---------------------------------------------------------

    corpus_text = "\n".join(sentences)
    corpus_hash = hashlib.sha256(
        corpus_text.encode("utf-8")
    ).hexdigest()

    print("\n--- Hash del corpus ---")
    print(f"Hash: {corpus_hash}")

    # ---------------------------------------------------------
    # GENERACIÓ DELS EMBEDDINGS
    # ---------------------------------------------------------

    print("\nGenerant embeddings...")

    embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings_list = embeddings.tolist()

    print(f"Embeddings generats: {len(embeddings_list)}")
    print(f"Dimensions de cada embedding: {embeddings.shape[1]}")

    # ---------------------------------------------------------
    # MOSTRA D'EMBEDDINGS
    # ---------------------------------------------------------

    print("\n--- Mostra d'embeddings ---")

    sample_indices = [0, 100, 1000]

    for index in sample_indices:
        sentence_id = ids[index]
        sentence = sentences[index]
        embedding = embeddings[index]

        print(f"\nID {sentence_id}")
        print(f"Frase: {sentence}")
        print(f"Embedding[:5]: {embedding[:5]}")
        print(f"Norma: {np.linalg.norm(embedding):.8f}")

    # ---------------------------------------------------------
    # ACTUALITZACIÓ DELS EMBEDDINGS A CHROMA
    # ---------------------------------------------------------

    times = []

    print("\nActualitzant embeddings a Chroma...")

    for doc_id, emb in zip(ids, embeddings_list):

        start = time.perf_counter()

        col_l2.update(
            ids=[doc_id],
            embeddings=[emb]
        )

        end = time.perf_counter()

        # També actualitzem la col·lecció amb distància cosine
        col_cos.update(
            ids=[doc_id],
            embeddings=[emb]
        )

        elapsed = end - start
        times.append(elapsed)

        if int(doc_id) % 1000 == 0:
            print(f"Actualitzats: {doc_id}/{len(sentences)}")

    # ---------------------------------------------------------
    # GUARDAR TEMPS
    # ---------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "sentence_id",
            "time_seconds"
        ])

        for doc_id, elapsed in zip(ids, times):
            writer.writerow([
                doc_id,
                elapsed
            ])

    # ---------------------------------------------------------
    # ESTADÍSTIQUES
    # ---------------------------------------------------------

    print("\n--- Estadístiques de temps d'actualització d'embeddings ---")

    print(f"Mínim: {min(times):.6f} segons")
    print(f"Màxim: {max(times):.6f} segons")
    print(f"Mitjana: {statistics.mean(times):.6f} segons")
    print(f"Desviació estàndard: {statistics.stdev(times):.6f} segons")

    print("\nC1 finalitzat correctament.")
    print(f"Temps guardats a: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
