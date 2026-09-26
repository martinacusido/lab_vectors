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
OUTPUT_FILE = "results/pgvector_query_times.csv"

# Els mateixos 10 IDs establerts a P2 i C2
QUERY_IDS = [101, 1101, 2101, 3101, 4101, 5101, 6101, 7101, 8101, 9101]

def main():
    print("Connectant amb PostgreSQL (Pgvector)...")
    conn = psycopg.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password= DB_PASSWORD
    )

    # Diccionaris per acumular els temps de cada mètrica per separat
    metric_times = {
        "euclidean": [],
        "cosine": []
    }

    results = []
    print("Començant les consultes G2...")

    for query_id in QUERY_IDS:
        for metric in ["euclidean", "cosine"]:
            print(f"Consulta ID {query_id} - {metric}")
            
            # Selecció de l'operador natiu de pgvector
            if metric == "euclidean":
                operator = "<->"
            else:
                operator = "<=>"
                
            with conn.cursor() as cur:
                start = time.perf_counter()
                cur.execute(
                    f"""
                    SELECT 
                        s.id, 
                        s.text, 
                        (s.embedding {operator} target.embedding) AS distance,
                        target.text AS target_text
                    FROM sentences_pgvector AS s
                    CROSS JOIN (
                        SELECT text, embedding 
                        FROM sentences_pgvector 
                        WHERE id = %s
                    ) AS target
                    WHERE s.id <> %s 
                      AND s.embedding IS NOT NULL
                    ORDER BY distance
                    LIMIT 2;
                    """,
                    (query_id, query_id)
                )
                nearest = cur.fetchall()
                end = time.perf_counter()
                
                elapsed = end - start
                metric_times[metric].append(elapsed)
                query_text = nearest[0][3]
                
                results.append([
                    query_id, query_text, metric, elapsed,
                    nearest[0][0], nearest[0][1], nearest[0][2],
                    nearest[1][0], nearest[1][1], nearest[1][2]
                ])
                
                print(
                    f"  Temps: {elapsed:.6f} s | "
                    f"Top 1: ID {nearest[0][0]} | "
                    f"Top 2: ID {nearest[1][0]}"
                )

    conn.close()

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_id", "query_text", "metric", "time_seconds",
            "top1_id", "top1_text", "top1_distance",
            "top2_id", "top2_text", "top2_distance"
        ])
        writer.writerows(results)

    print("\n--- Estadístiques de temps de consulta ---")
    for metric, times_list in metric_times.items():
        print(f"\nMètrica: {metric.upper()}")
        print(f"Mínim: {min(times_list):.6f} segons")
        print(f"Màxim: {max(times_list):.6f} segons")
        print(f"Mitjana: {statistics.mean(times_list):.6f} segons")
        print(f"Desviació estàndard: {statistics.stdev(times_list):.6f} segons")

    print("\nG2 finalitzat correctament.")
    print(f"Resultats guardats a: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()