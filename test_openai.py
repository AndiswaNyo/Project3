import os
from dotenv import load_dotenv
from openai import OpenAI

print("Loading .env ...")
load_dotenv()

print(".env present?", os.path.exists(".env"))
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

print("API key loaded?", bool(api_key))
print("Model:", model)

try:
    client = OpenAI(api_key=api_key, timeout=30)
    print("Sending request ...")
    resp = client.responses.create(
        model=model,
        input="Say hello in one short sentence."
    )
    print("Response:", resp.output_text)
except Exception as e:
    print("ERROR:", repr(e))