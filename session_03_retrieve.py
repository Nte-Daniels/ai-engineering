import os
from dotenv import load_dotenv
import psycopg2
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

# Connect to database
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="postgres",
    user="postgres",
    password="postgres"
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Connected and models loaded.")

def retrieve_chunks(query, top_k=3):
    query_embedding = model.encode(query).tolist()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT content, filename, chunk_index
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, top_k))
    results = cursor.fetchall()
    cursor.close()
    return results

def ask(question):
    chunks = retrieve_chunks(question)
    context = "\n\n".join([chunk[0] for chunk in chunks])
    
    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context only."},
            {"role": "user", "content": prompt}
        ]
    )
    
    answer = response.choices[0].message.content
    tokens = response.usage
    
    return answer, chunks, tokens

# Test it
question = "What is the difference between 1NF and 2NF?"
answer, chunks, tokens = ask(question)

print(f"\nQuestion: {question}")
print(f"\nAnswer: {answer}")
print(f"\nRetrieved from:")
for chunk in chunks:
    print(f"  - {chunk[1]} (chunk {chunk[2]})")
print(f"\nTokens used - Input: {tokens.prompt_tokens}, Output: {tokens.completion_tokens}")