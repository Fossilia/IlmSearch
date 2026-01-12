import httpx
import asyncio

# Mapping specific book names to the API's edition format
BOOK_MAPPINGS = {
    "bukhari": {"eng": "eng-bukhari", "ara": "ara-bukhari"},
    "muslim":  {"eng": "eng-muslim",  "ara": "ara-muslim"},
    "abudawud": {"eng": "eng-abudawud", "ara": "ara-abudawud"},
    "ibnmajah": {"eng": "eng-ibnmajah", "ara": "ara-ibnmajah"},
    "tirmidhi": {"eng": "eng-tirmidhi", "ara": "ara-tirmidhi"},
    "nasai":    {"eng": "eng-nasai",    "ara": "ara-nasai"}
}

async def fetch_hadith(book: str, number: int):
    """
    Fetches English and Arabic text for a specific Hadith.
    """
    clean_book = book.lower().strip()
    
    if clean_book not in BOOK_MAPPINGS:
        return {"error": f"Book '{book}' not found. Available: {list(BOOK_MAPPINGS.keys())}"}

    editions = BOOK_MAPPINGS[clean_book]
    base_url = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions"
    
    # Construct URLs
    eng_url = f"{base_url}/{editions['eng']}/{number}.json"
    ara_url = f"{base_url}/{editions['ara']}/{number}.json"

    async with httpx.AsyncClient() as client:
        try:
            # Fetch both languages at the same time
            eng_resp, ara_resp = await asyncio.gather(
                client.get(eng_url),
                client.get(ara_url)
            )

            if eng_resp.status_code != 200:
                return {"error": "Hadith not found in external API."}

            eng_data = eng_resp.json()
            ara_data = ara_resp.json() if ara_resp.status_code == 200 else {}

            return {
                "id": f"{clean_book}:{number}",
                "book": clean_book.capitalize(),
                "number": number,
                "english": eng_data.get("text", "") or eng_data.get("hadith", ""),
                "arabic": ara_data.get("text", "") or ara_data.get("hadith", ""),
                "grade": eng_data.get("grades", []),
                "error": None
            }
        except Exception as e:
            return {"error": str(e)}