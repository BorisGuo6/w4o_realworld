from google_genai import genai
from google_genai.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Explain how AI works in a few words",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
    ),
)
print(response.text)