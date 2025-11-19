import json
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# Load Qur'an dataset
with open("quran_dataset.json", "r") as f:
    quran = json.load(f)

# Load FAISS index
index = faiss.read_index("quran.index")

# Embedding function
def embed(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")


def search(query: str, k: int = 3):
    print(f"\nSearching for: {query}\n")

    # Embed query
    q_vec = embed(query)

    # Search FAISS
    distances, idxs = index.search(np.array([q_vec]), k)

    # Display results
    for rank, i in enumerate(idxs[0]):
        item = quran[i]
        print(f"{rank+1}. {item['ref']}")
        print(f"   {item['text']}")
        print()

# Example usage
if __name__ == "__main__":
    user_query = input("Enter your question: ")
    search(user_query)