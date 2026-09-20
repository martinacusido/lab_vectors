import csv
import time
import chromadb
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
    ids = data['ids']
    sentences = data['documents']
    print(f"Frases llegides de Chroma: {len(sentences)}")

    # Chroma no garanteix l'ordre al fer get(), així que ho ordenem numèricament
    sorted_data = sorted(zip(ids, sentences), key=lambda x: int(x[0]))
    ids, sentences = zip(*sorted_data)

    print("Generant embeddings...")
    embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)
    embeddings_list = embeddings.tolist()
    print(f"Embeddings generats: {len(embeddings_list)}")

    times = []
    print("Actualitzant embeddings a Chroma...")
    for doc_id, emb in zip(ids, embeddings_list):
        start = time.perf_counter()
        col_l2.update(ids=[doc_id], embeddings=[emb])
        end = time.perf_counter()
        
        col_cos.update(ids=[doc_id], embeddings=[emb])
        
        elapsed = end - start
        times.append(elapsed)

        if int(doc_id) % 1000 == 0:
            print(f"Actualitzats: {doc_id}/{len(sentences)}")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence_id", "time_seconds"])
        for doc_id, elapsed in zip(ids, times):
            writer.writerow([doc_id, elapsed])

    print("\nC1 finalitzat correctament.")
    print(f"Temps guardats a: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()