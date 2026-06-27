from sentence_transformers import SentenceTransformer

import os
from dotenv import load_dotenv
import psycopg2
from pypdf import PdfReader

load_dotenv()

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="postgres",
    user="postgres",
    password="postgres"
)

print("Connected to PostgreSQL successfully")

cursor = conn.cursor()

cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename TEXT,
    chunk_index INTEGER,
    content TEXT,
    embedding vector(384)
);
""")

conn.commit()
print("Database ready.")

# Load and chunk PDFs
def extract_text_from_pdf(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks

# Process all PDFs in data folder
data_folder = "data"
all_chunks = []

for filename in os.listdir(data_folder):
    if filename.endswith(".pdf"):
        filepath = os.path.join(data_folder, filename)
        print(f"Processing: {filename}")
        text = extract_text_from_pdf(filepath)
        chunks = chunk_text(text)
        print(f"  Extracted {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            all_chunks.append((filename, i, chunk))

print(f"\nTotal chunks across all documents: {len(all_chunks)}")

# Load embedding model
print("\nLoading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# Embed and store chunks
cursor = conn.cursor()

print("Embedding and storing chunks...")
for filename, chunk_index, content in all_chunks:
    embedding = model.encode(content).tolist()
    cursor.execute(
        "INSERT INTO documents (filename, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
        (filename, chunk_index, content, embedding)
    )

conn.commit()
print(f"Stored {len(all_chunks)} chunks in pgvector.")

cursor.close()
conn.close()
print("Ingestion complete.")