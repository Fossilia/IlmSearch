import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------------
# Load Quran metadata
# --------------------------------
with open("quran_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

lookup = {}
for v in metadata:
    key = f"{v['surah_id']}:{v['verse_id']}"
    lookup[key] = v


def get_verse_indexes(query: str, k: int = 5):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Return ONLY a JSON object with this format:\n"
                    "{ \"results\": [ {\"ref\": \"surah:ayah\"}, ... ] }\n\n"
                    "Rules:\n"
                    "- No commentary\n"
                    "- No explanations\n"
                    "- No trailing text\n"
                    "- Only valid JSON\n"
                    "- If unsure, guess the best possible related verses"
                )
            },
            {
                "role": "user",
                "content": f"Return the top {k} Quran verse references for this query: {query}"
            },
        ]
    )

    content = response.choices[0].message.content
    try:
        data = json.loads(content)
        return data.get("results", [])
    except Exception:
        print("JSON parsing failed. Raw output:")
        print(content)
        return []

# --------------------------------
# Main search
# --------------------------------
def search(query: str, k: int = 5):
    print(f"\nQuery: {query}")

    refs = get_verse_indexes(query, k)

    print("\nModel returned:")
    print(refs)

    print("\n--- RESULTS ---\n")

    for item in refs:
        ref = item.get("ref")
        if not ref:
            print("Invalid ref:", item)
            continue

        if ref in lookup:
            verse = lookup[ref]
            print(f"{verse['surah_name']} {verse['verse_id']}")
            print(f"Arabic:  {verse['text_ar']}")
            print(f"English: {verse['text_en']}\n")
        else:
            print(f"{ref} → Not found locally.\n")


if __name__ == "__main__":
    user_query = input("Enter your question: ")
    search(user_query)
