import csv
import time
import statistics
import chromadb

OUTPUT_FILE = "results/chroma_query_times.csv"

# Els mateixos IDs exactes utilitzats a P2 per fer justa la comparació
QUERY_IDS = ['101', '1101', '2101', '3101', '4101', '5101', '6101', '7101', '8101', '9101']

def main():
    print("Connectant amb Chroma...")
    client = chromadb.PersistentClient(path="./chroma_db")
    col_l2 = client.get_collection("sentences_l2")
    col_cos = client.get_collection("sentences_cosine")
    
    results = []
    
    # Diccionaris per acumular els temps de cada mètrica per separat
    metric_times = {
        "euclidean": [],
        "cosine": []
    }
    
    print("Començant les consultes C2...")
    
    for query_id in QUERY_IDS:
        # Obtenim l'embedding objectiu prèviament calculat i també el seu text
        target_data = col_l2.get(ids=[query_id], include=["embeddings", "documents"])
        target_emb = target_data['embeddings'][0]
        target_text = target_data['documents'][0]
        
        # Iterem per les dues col·leccions per obtenir les dues mètriques demanades
        for metric, col in [("euclidean", col_l2), ("cosine", col_cos)]:
            print(f"Consulta ID {query_id} - {metric}")
            
            start = time.perf_counter()
            # Demanem 3 resultats perquè la pròpia frase objectiu serà el resultat #1 (distància 0)
            search_results = col.query(
                query_embeddings=[target_emb],
                n_results=3,
                include=["distances", "documents"]
            )
            end = time.perf_counter()
            
            elapsed = end - start
            metric_times[metric].append(elapsed) # Guardem el temps per a l'estadística
            
            res_ids = search_results['ids'][0]
            res_dists = search_results['distances'][0]
            res_texts = search_results['documents'][0]
            
            # Filtrem el propi ID perquè l'enunciat demana "entre totes les ALTRES frases"
            filtered = [(i, d, t) for i, d, t in zip(res_ids, res_dists, res_texts) if i != query_id]
            
            top1_id, top1_dist, top1_text = filtered[0]
            top2_id, top2_dist, top2_text = filtered[1]
            
            results.append([
                query_id,
                target_text,
                metric,
                elapsed,
                top1_id,
                top1_text,
                top1_dist,
                top2_id,
                top2_text,
                top2_dist
            ])
            print(f"  Temps: {elapsed:.6f} s | Top 1: ID {top1_id} | Top 2: ID {top2_id}")

    # Impressió de les estadístiques per cada mètrica
    print("\n--- Estadístiques de temps de consulta ---")
    for metric, times_list in metric_times.items():
        print(f"\nMètrica: {metric.upper()}")
        print(f"Mínim: {min(times_list):.6f} segons")
        print(f"Màxim: {max(times_list):.6f} segons")
        print(f"Mitjana: {statistics.mean(times_list):.6f} segons")
        print(f"Desviació estàndard: {statistics.stdev(times_list):.6f} segons")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
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
        
    print("\nC2 finalitzat correctament.")
    print(f"Resultats guardats a: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()