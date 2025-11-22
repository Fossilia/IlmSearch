import json
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

with open("quran_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

index = faiss.read_index("quran.index")

def embed(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")


def search(query: str, k: int = 5):
    print(f"\nSearching for: {query}\n")

    q_vec = embed(query)

    # Search FAISS
    distances, idxs = index.search(np.array([q_vec]), k)

    # Display results
    for rank, idx in enumerate(idxs[0]):
        verse = metadata[idx]

        print(f"{rank+1}. {verse['surah_name']} {verse['verse_id']}")
        print(f"   Arabic:  {verse['text_ar']}")
        print(f"   English: {verse['text_en']}")
        #print("   Verse object:", json.dumps(verse, ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    user_query = input("Enter your question: ")
    search(user_query)