import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a senior data engineer. Be concise. Use technical language. Never exceed 2 sentences."},
        {"role": "user", "content": "What is Apache Spark?"}
    ]
)

print(response.choices[0].message.content)
print(response.usage)
print("---")
print("Model used:", response.model)
print("Stop reason:", response.choices[0].finish_reason)
print("Input tokens:", response.usage.prompt_tokens)
print("Output tokens:", response.usage.completion_tokens)
print("Total cost tokens:", response.usage.total_tokens)

# TODO: What happens if the model ignores the system prompt entirely?
# How would you detect that programmatically?