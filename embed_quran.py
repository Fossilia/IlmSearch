import json
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import os

# Load API Key
load_dotenv()
client = OpenAI()


# ---- Load Qur'an dataset ----
with open("quran_dataset.json", "r") as f:
    quran = json.load(f)


# ---- Embedding function ----
def embed(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")


# ---- Embed all ayahs ----
print("Embedding Qur'an dataset...")

all_embeddings = []
for item in quran:
    vec = embed(item["text"])
    all_embeddings.append(vec)

all_embeddings = np.vstack(all_embeddings)
dimension = all_embeddings.shape[1]


# ---- Create FAISS index ----
index = faiss.IndexFlatL2(dimension)
index.add(all_embeddings)

# ---- Save FAISS index ----
faiss.write_index(index, "quran.index")

print("Done. Saved FAISS index as quran.index")