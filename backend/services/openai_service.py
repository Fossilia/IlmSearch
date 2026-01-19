import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def fetch_refs_from_openai(query: str, k: int = 5):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Return ONLY a JSON object with this exact format:\n"
                    "{\n"
                    "  \"quran\": [ {\"ref\": \"surah:ayah\"}, ... ],\n"
                    "  \"hadith\": [ {\"ref\": \"book:number\"}, ... ]\n"
                    "}\n\n"
                    "Rules:\n"
                    "1. Quran: Use format 'surah:ayah' (e.g., \"2:255\"). No ranges.\n"
                    "2. Hadith: Use format 'book:number' (e.g., \"bukhari:1\").\n"
                    "   - Supported books: 'bukhari', 'muslim', 'nawawi40'.\n"
                    "   - Use Sunnah.com numbering.\n"
                    "   - 'bukhari': Use standard USC-MSA numbering (1-7563).\n"
                    "   - 'muslim': Use USC-MSA numbering (1-3033). WARNING: This often differs from standard Arabic numbering. If unsure, verify the USC-MSA ID.\n"
                    "   - 'nawawi40': MAX NUMBER IS 42.\n"
                    "3. No commentary, no explanations, only valid JSON."
                )
            },
            {
                "role": "user",
                "content": f"Return the top {k} Quran verses and top {k} Hadith references for: {query}"
            },
        ]
    )

    content = response.choices[0].message.content
    print("OpenAI Response Content:", content)

    try:
        data = json.loads(content)
        return {
            "quran": data.get("quran", []),
            "hadith": data.get("hadith", [])
        }
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        return {"quran": [], "hadith": [], "error": "JSON parsing failed"}