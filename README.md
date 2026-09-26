# Lab: On the Impedance Mismatch of Vector Data

Aquest repositori conté el codi per a la pràctica sobre el _impedance mismatch_ de dades vectorials, comparant el rendiment i l'ús d'una base de dades relacional tradicional (PostgreSQL), una base de dades nativa vectorial (ChromaDB) i una solució híbrida (Pgvector).

## 📂 Estructura del projecte

- `/data/`: Conté l'script per generar el corpus (`create_corpus.py`) i el fitxer de dades generat (`sentences.txt`).
- `/postgresql/`: Scripts per a l'avaluació amb PostgreSQL estàndard (`P0.py`, `P1.py`, `P2.py`).
- `/chroma/`: Scripts per a l'avaluació amb ChromaDB (`C0.py`, `C1.py`, `C2.py`).
- `/pg_vector/`: Scripts per a l'avaluació de la part opcional amb l'extensió Pgvector (`G0.py`, `G1.py`, `G2.py`).
- `/results/`: (Crear manualment) Carpeta on es desaran els fitxers `.csv` amb els temps d'execució.

---

## ⚙️ Requisits previs

1. **Python 3.8+** instal·lat al sistema.
2. **PostgreSQL** instal·lat i en execució.
3. Extensió **Pgvector** instal·lada a la instància de PostgreSQL (necessària per a la part opcional `G0`, `G1` i `G2`).

---

## 🚀 Configuració de l'entorn

### 1. Instal·lació de dependències

Es recomana utilitzar un entorn virtual. Obre un terminal a l'arrel del projecte i executa els següents comandaments per instal·lar les llibreries necessàries:

```bash
# Crear i activar entorn virtual (opcional però recomanat)
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# venv\Scripts\activate   # En Windows

# Instal·lar els paquets requerits
pip install psycopg sentence-transformers chromadb numpy python-dotenv
```

### 2. Configuració de la Base de Dades (.env)

Per evitar posar credencials directament al codi, utilitzem variables d'entorn.

1. A l'arrel del projecte, crea un fitxer anomenat `.env` copiant l'estructura de `.env.example` o crea'l directament amb el següent contingut:

```env
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=la_teva_contrasenya_aqui
```

_Substitueix `la_teva_contrasenya_aqui` per la contrasenya del teu usuari de PostgreSQL._

### 3. Crear el directori de resultats

Assegura't que existeix una carpeta anomenada `results` a l'arrel del projecte per tal que els scripts puguin guardar els fitxers `.csv` correctament:

```bash
mkdir results
```

---

## ▶️ Execució de les proves

Pots executar els diferents mòduls de manera independent, però és important seguir l'ordre numèric dins de cada mòdul (0 -> 1 -> 2) per garantir que les dades s'insereixen abans de ser consultades.

Tots els scripts s'han d'executar **des de l'arrel del projecte**.

### Opció A: PostgreSQL Clàssic

```bash
python postgresql/P0.py  # Inserció de text
python postgresql/P1.py  # Generació i inserció d'embeddings
python postgresql/P2.py  # Consulta del top-2 de frases similars
```

### Opció B: ChromaDB (Vector Database)

_Nota: Chroma guardarà la base de dades localment en una carpeta anomenada `chroma_db` a l'arrel del projecte._

```bash
python chroma/C0.py      # Inserció de text
python chroma/C1.py      # Generació i actualització d'embeddings
python chroma/C2.py      # Consulta del top-2 de frases similars
```

### Opció C: Pgvector (Solució híbrida)

```bash
python pg_vector/G0.py   # Inserció de text (crea la taula amb extensió vector)
python pg_vector/G1.py   # Generació i inserció d'embeddings
python pg_vector/G2.py   # Consulta mitjançant operadors natius (<-> i <=>)
```

---

## 📊 Resultats

Un cop executats, trobaràs tots els temps mesurats (en segons) en format CSV dins de la carpeta `/results/`. Aquests fitxers contenen els temps d'inserció de text, inserció d'embeddings i consultes KNN, separats per cada tecnologia analitzada.
