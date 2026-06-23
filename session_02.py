import json

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

client = OpenAI()


class PipelineExplanation(BaseModel):
    definition: str
    use_case: str
    tools: list[str]
    complexity: str
    author: str

print(PipelineExplanation.model_fields)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a senior data engineer. Return valid JSON only. No extra text."},
        {"role": "user", "content": """
Explain what a data pipeline is.

Return JSON with:
- definition
- use_case
- tools (list of 3 tools)
- complexity (must be low, medium, or high)

Return valid JSON only.
"""}
    ]
)

raw = response.choices[0].message.content

print("Raw output:")
print(raw)


cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

try:
    parsed = PipelineExplanation(**json.loads(cleaned))
    print("\nParsed and validated:")
    print("Definition:", parsed.definition)
    print("Use case:", parsed.use_case)
    print("Tools:", parsed.tools)
    print("Complexity:", parsed.complexity)
except Exception as e:
    print("\nValidation failed:", e)