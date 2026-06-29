import os
from dotenv import load_dotenv
import psycopg2
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

# Connect to database
try:
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="postgres",
        user="postgres",
        password="postgres"
    )
    print("Connected to PostgreSQL successfully")
except Exception as e:
    print(f"Error connecting to PostgreSQL: {e}")
    exit()

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

    print(f"Retrieved {len(chunks)} chunks for the question: '{question}'")

    if not chunks:
        print("No chunks were retrieved from the database.")
        return "I don't have that information.", [], None

    context = "\n\n".join([chunk[0] for chunk in chunks])
    
    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided context only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content

        if "I don't have that information" in answer:
            print("Evaluation: No answer found in the retrieved context.")
        else:
            print("Evaluation: Answer successfully generated from retrieved context.")

        tokens = response.usage

        return answer, chunks, tokens

    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        return None, chunks, None

# Test it
# Test questions for evaluation
questions = [
    "What is data cleaning and why is it important?",
    "How do you remove duplicates in Power Query?",
    "What is a bridge table and when do you use it?"
]

# Question 3 is expected to fail because the vector database
# does not contain information about bridge tables or normalization.
# Therefore, the model should return:
# "I don't have that information."

for question in questions:

    answer, chunks, tokens = ask(question)

    print(f"\nQuestion: {question}")
    if answer:
        print(f"\nAnswer: {answer}")
    else:
        print("\nNo answer returned.")

    print("\nRetrieved from:")
    for chunk in chunks:
        print(f"  - {chunk[1]} (chunk {chunk[2]})")

    if tokens:
        print(f"\nTokens used - Input: {tokens.prompt_tokens}, Output: {tokens.completion_tokens}")
    print("\n" + "=" * 60)