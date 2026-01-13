import json
from pathlib import Path

# Load metadata once
metadata_path = Path(__file__).resolve().parents[1] / "data" / "quran_metadata.json"
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Build lookups
lookup = {}
for v in metadata:
    key = f"{v['surah_id']}:{v['verse_id']}"
    lookup[key] = v

def get_quran_verse_by_ref(ref: str):
    """
    Get a verse by its reference string (e.g., '1:1').
    """
    if ref in lookup:
        verse = lookup[ref]
        return {
            "reference": ref,
            "surah_name": verse["surah_name"],
            "verse_id": verse["verse_id"],
            "arabic": verse["text_ar"],
            "english": verse["text_en"],
        }
    else:
        return {"error": "Verse not found locally"}

def get_quran_verses_by_refs(refs: list):
    """
    Get verses by a list of reference dicts.
    """
    results = []
    for item in refs:
        ref = item.get("ref")
        if not ref:
            continue
        result = get_quran_verse_by_ref(ref)
        results.append(result)
    return results
