import json
import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


with open("quran_dataset.json", "r", encoding="utf-8") as f:
    surahs = json.load(f)

texts = []        # strings to embed
metadata = []     # reference info

for surah in surahs:
    surah_id = surah["id"]
    surah_name = surah["transliteration"]

    for verse in surah["verses"]:
        english = verse.get("translation", "")
        arabic = verse.get("text", "")

        # Skip empty/invalid 
        if not english or not isinstance(english, str):
            continue

        # Combine for better semantic retrieval
        combined_text = f"{english}\n{arabic}"

        ref = f"{surah_name} {verse['id']}"

        texts.append(combined_text)
        metadata.append({
            "surah_id": surah_id,
            "surah_name": surah_name,
            "verse_id": verse["id"],
            "text_en": english,
            "text_ar": arabic,
            "ref": ref
        })

print(f"Loaded {len(texts)} valid verses for embedding.")

# --------------------------------
# Generate embeddings (batched)
# --------------------------------
def embed_batch(batch):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=batch  # MUST be list of strings
    )
    return [d.embedding for d in response.data]


all_embeddings = []
batch_size = 128

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    print(f"Embedding batch {i//batch_size + 1} ...")
    batch_embeds = embed_batch(batch)
    all_embeddings.extend(batch_embeds)

embeddings = np.array(all_embeddings, dtype="float32")
dimension = embeddings.shape[1]

print(f"Embedding complete. Shape: {embeddings.shape}")

# --------------------------------
# Create FAISS index
# --------------------------------
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"FAISS index created with {index.ntotal} vectors.")

# --------------------------------
# Save index + metadata
# --------------------------------
faiss.write_index(index, "quran.index")

with open("quran_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("Saved quran.index and quran_metadata.json")