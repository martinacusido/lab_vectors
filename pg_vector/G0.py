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
OUTPUT_FILE = "results/pgvector_text_times.csv"

def main():
    # Connectar amb PostgreSQL
    conn = psycopg.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password= DB_PASSWORD
    )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    print(f"Frases llegides: {len(sentences)}")

    times = []
    with conn:
        with conn.cursor() as cur:
            # Habilitem l'extensió pgvector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Creem la taula específica per a pgvector (amb 384 dimensions)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sentences_pgvector (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding vector(384)
                );
            """)
            
            cur.execute("TRUNCATE TABLE sentences_pgvector RESTART IDENTITY;")
            
            for i, sentence in enumerate(sentences, start=1):
                start = time.perf_counter()
                cur.execute(
                    "INSERT INTO sentences_pgvector (text) VALUES (%s)",
                    (sentence,)
                )
                end = time.perf_counter()
                times.append(end - start)
                
                if i % 1000 == 0:
                    print(f"Inserides: {i}/{len(sentences)}")
    conn.close()

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

    print("\nG0 finalitzat correctament.")
    print(f"Temps guardats a: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()