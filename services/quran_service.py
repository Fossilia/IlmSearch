import json
from pathlib import Path
from .openai_service import get_verse_indexes

# Load metadata once
metadata_path = Path(__file__).resolve().parents[1] / "data" / "quran_metadata.json"
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Build lookup
lookup = {}
for v in metadata:
    key = f"{v['surah_id']}:{v['verse_id']}"
    lookup[key] = v


def search(query: str, k: int = 5):
    refs = get_verse_indexes(query, k)
    results = []

    for item in refs:
        ref = item.get("ref")
        if not ref:
            continue

        if ref in lookup:
            verse = lookup[ref]
            results.append({
                "reference": ref,
                "surah_name": verse["surah_name"],
                "verse_id": verse["verse_id"],
                "arabic": verse["text_ar"],
                "english": verse["text_en"],
            })
        else:
            results.append({
                "reference": ref,
                "error": "Not found locally"
            })

    return results
