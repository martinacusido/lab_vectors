import csv
import time
import psycopg

DB_NAME = "vector_lab"
DB_USER = "propietario"

INPUT_FILE = "data/sentences.txt"
OUTPUT_FILE = "results/postgresql_text_times.csv"


def main():
    # Connectar amb PostgreSQL
    conn = psycopg.connect(
        dbname=DB_NAME,
        user=DB_USER
    )

    # Llegir les frases
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"Frases llegides: {len(sentences)}")

    times = []

    with conn:
        with conn.cursor() as cur:

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

    print()
    print("P0 finalitzat correctament.")
    print(f"Temps guardats a: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
