import json
import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------
# Load FAISS index
# -------------------------------------------------
index = faiss.read_index("quran.index")

# -------------------------------------------------
# Load metadata
# -------------------------------------------------
with open("quran_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)


# -------------------------------------------------
# Embedding function for queries
# -------------------------------------------------
def embed(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")


# -------------------------------------------------
# Main search function
# -------------------------------------------------
def search(query: str, k: int = 5):
    print(f"\nSearching for: {query}\n")

    # Embed query
    q_vec = embed(query)

    # Perform FAISS search
    distances, idxs = index.search(np.array([q_vec]), k)

    # Show results
    for rank, i in enumerate(idxs[0]):
        verse = metadata[i]

        print(f"{rank+1}. {verse['ref']}")
        print(f"   English: {verse['text_en']}")
        print(f"   Arabic:  {verse['text_ar']}")
        print()


# -------------------------------------------------
# Run interactively
# -------------------------------------------------
if __name__ == "__main__":
    user_query = input("Enter your question: ")
    search(user_query)
