import csv
import time
import chromadb

INPUT_FILE = "data/sentences.txt"
OUTPUT_FILE = "results/chroma_text_times.csv"

def main():
    print("Connectant amb Chroma...")
    # ChromaPersistentClient guarda les dades en disc a la carpeta ./chroma_db
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Eliminem col·leccions si existien prèviament per a una execució neta
    try:
        client.delete_collection("sentences_l2")
        client.delete_collection("sentences_cosine")
    except:
        pass
        
    # Creem dues col·leccions per a les dues mètriques de distància de l'enunciat
    col_l2 = client.create_collection(name="sentences_l2", metadata={"hnsw:space": "l2"})
    col_cos = client.create_collection(name="sentences_cosine", metadata={"hnsw:space": "cosine"})

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    print(f"Frases llegides: {len(sentences)}")

    times = []
    # Vector fals per forçar a Chroma a guardar ràpid sense usar el seu model per defecte
    dummy_embedding = [0.0] * 384

    print("Inserint text a Chroma...")
    for i, sentence in enumerate(sentences, start=1):
        doc_id = str(i)
        
        start = time.perf_counter()
        col_l2.add(ids=[doc_id], documents=[sentence], embeddings=[dummy_embedding])
        end = time.perf_counter()
        
        # També inserim a la col·lecció cosine per tenir-la a punt per a C1 i C2
        col_cos.add(ids=[doc_id], documents=[sentence], embeddings=[dummy_embedding])
        
        elapsed = end - start
        times.append(elapsed)
        
        if i % 1000 == 0:
            print(f"Inserides: {i}/{len(sentences)}")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence_id", "time_seconds"])
        for i, elapsed in enumerate(times, start=1):
            writer.writerow([i, elapsed])

    print("\nC0 finalitzat correctament.")
    print(f"Temps guardats a: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()