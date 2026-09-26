import csv
import os
import statistics
import time
import psycopg
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

INPUT_FILE = "data/sentences.txt"
OUTPUT_FILE = "results/postgresql_text_times.csv"


def main():
    # Connectar amb PostgreSQL
    conn = psycopg.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password= DB_PASSWORD
    )

    # Llegir les frases
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"Frases llegides: {len(sentences)}")

    times = []

    with conn:
        with conn.cursor() as cur:

            cur.execute("""
                CREATE TABLE IF NOT EXISTS sentences (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding REAL[]
                );
            """)

            # Buidem la taula abans de començar
            cur.execute("TRUNCATE TABLE sentences RESTART IDENTITY;")

            for i, sentence in enumerate(sentences, start=1):

                start = time.perf_counter()

                cur.execute(
                    "INSERT INTO sentences (text) VALUES (%s)",
                    (sentence,)
                )

                end = time.perf_counter()

                elapsed = end - start
                times.append(elapsed)

                if i % 1000 == 0:
                    print(f"Inserides: {i}/{len(sentences)}")

    conn.close()

    # Guardar els temps
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence_id", "time_seconds"])

        for i, elapsed in enumerate(times, start=1):
            writer.writerow([i, elapsed])

    print("\n--- Estadístiques de temps d'inserció ---")
    print(f"Mínim: {min(times):.6f} segons")
    print(f"Màxim: {max(times):.6f} segons")
    print(f"Mitjana: {statistics.mean(times):.6f} segons")
    print(f"Desviació estàndard: {statistics.stdev(times):.6f} segons")
    
    print("P0 finalitzat correctament.")
    print(f"Temps guardats a: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
