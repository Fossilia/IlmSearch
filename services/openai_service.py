import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_verse_indexes(query: str, k: int = 5):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Return ONLY a JSON object with this format (surah and ayah are numbers):\n"
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
        return json.loads(content).get("results", [])
    except Exception:
        return {"error": "JSON parsing failed", "raw_output": content}
