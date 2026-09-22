import csv
import time
import psycopg

DB_NAME = "vector_lab"
DB_USER = "propietario"
OUTPUT_FILE = "results/postgresql_query_times.csv"

QUERY_IDS = [
    101, 1101, 2101, 3101, 4101,
    5101, 6101, 7101, 8101, 9101
]

def main():
    print("Connectant amb PostgreSQL...")
    conn = psycopg.connect(
        dbname=DB_NAME,
        user=DB_USER
    )
    
    with conn.cursor() as cur:
        print("Preparant les funcions de distància...")
        cur.execute("""
            CREATE OR REPLACE FUNCTION euclidean_distance(
                a real[],
                b real[]
            )
            RETURNS double precision
            AS $$
                SELECT sqrt(
                    sum(
                        power(
                            a[i]::double precision
                            - b[i]::double precision,
                            2
                        )
                    )
                )
                FROM generate_subscripts(a, 1) AS s(i);
            $$
            LANGUAGE SQL
            IMMUTABLE;
        """)
        
        cur.execute("""
            CREATE OR REPLACE FUNCTION cosine_distance(
                a real[],
                b real[]
            )
            RETURNS double precision
            AS $$
                SELECT 1.0 -
                    (
                        sum(
                            a[i]::double precision
                            * b[i]::double precision
                        )
                        /
                        (
                            sqrt(
                                sum(
                                    a[i]::double precision
                                    * a[i]::double precision
                                )
                            )
                            *
                            sqrt(
                                sum(
                                    b[i]::double precision
                                    * b[i]::double precision
                                )
                            )
                        )
                    )
                FROM generate_subscripts(a, 1) AS s(i);
            $$
            LANGUAGE SQL
            IMMUTABLE;
        """)
    conn.commit()
    print("Funcions preparades.\n")
    
    results = []
    print("Començant les consultes P2...")
    
    for query_id in QUERY_IDS:
        for metric in ["euclidean", "cosine"]:
            print(f"Consulta ID {query_id} - {metric}")
            if metric == "euclidean":
                function = "euclidean_distance"
            else:
                function = "cosine_distance"
                
            with conn.cursor() as cur:
                start = time.perf_counter()
                cur.execute(
                    f"""
                    SELECT
                        s.id,
                        s.text,
                        {function}(s.embedding, target.embedding) AS distance,
                        target.text AS target_text
                    FROM sentences AS s
                    CROSS JOIN (
                        SELECT text, embedding
                        FROM sentences
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
            
            # nearest[0][0] = s.id, nearest[0][1] = s.text, 
            # nearest[0][2] = distance, nearest[0][3] = target_text
            query_text = nearest[0][3]
            
            results.append([
                query_id,
                query_text,
                metric,
                elapsed,
                nearest[0][0],
                nearest[0][1],
                nearest[0][2],
                nearest[1][0],
                nearest[1][1],
                nearest[1][2]
            ])
            
            print(
                f"  Temps: {elapsed:.6f} s | "
                f"Top 1: ID {nearest[0][0]} | "
                f"Top 2: ID {nearest[1][0]}"
            )
            
    conn.close()
    
    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_id",
            "query_text",
            "metric",
            "time_seconds",
            "top1_id",
            "top1_text",
            "top1_distance",
            "top2_id",
            "top2_text",
            "top2_distance"
        ])
        writer.writerows(results)
        
    print("\nP2 finalitzat correctament.")
    print(f"Resultats guardats a: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()