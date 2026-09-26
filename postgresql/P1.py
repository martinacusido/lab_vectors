import csv
import os
import time
import hashlib
import statistics
import numpy as np
import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

OUTPUT_FILE = "results/postgresql_embedding_times.csv"


def main():
    print("Carregant el model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connectant amb PostgreSQL...")
    conn = psycopg.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password= DB_PASSWORD
    )

    # Llegim les frases i els seus IDs
    with conn.cursor() as cur:
        cur.execute("SELECT id, text FROM sentences ORDER BY id")
        rows = cur.fetchall()

    ids = [row[0] for row in rows]
    sentences = [row[1] for row in rows]

    print(f"Frases llegides: {len(sentences)}")

    # ---------------------------------------------------------
    # GENERACIÓ DELS EMBEDDINGS
    # ---------------------------------------------------------

    print("\nGenerant embeddings...")

    embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    print(f"Embeddings generats: {len(embeddings)}")
    print(f"Dimensions de cada embedding: {embeddings.shape[1]}")


    # ---------------------------------------------------------
    # GUARDAR ELS EMBEDDINGS A POSTGRESQL
    # ---------------------------------------------------------

    times = []

    print("\nGuardant embeddings a PostgreSQL...")

    with conn:
        with conn.cursor() as cur:

            for i, (sentence_id, embedding) in enumerate(
                zip(ids, embeddings), start=1
            ):
                start = time.perf_counter()

                cur.execute(
                    "UPDATE sentences SET embedding = %s WHERE id = %s",
                    (
                        embedding.astype(np.float32).tolist(),
                        sentence_id
                    )
                )

                end = time.perf_counter()

                elapsed = end - start
                times.append(elapsed)

                if i % 1000 == 0:
                    print(f"Guardats: {i}/{len(sentences)}")

    conn.close()

    # ---------------------------------------------------------
    # GUARDAR ELS TEMPS
    # ---------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)
        writer.writerow(["sentence_id", "time_seconds"])
        for doc_id, elapsed in zip(ids, times):
            writer.writerow([doc_id, elapsed])

    print("\n--- Estadístiques de temps d'actualització d'embeddings ---")
    print(f"Mínim: {min(times):.6f} segons")
    print(f"Màxim: {max(times):.6f} segons")
    print(f"Mitjana: {statistics.mean(times):.6f} segons")
    print(f"Desviació estàndard: {statistics.stdev(times):.6f} segons")
    
    print("P1 finalitzat correctament.")
    print(f"Temps guardats a: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
