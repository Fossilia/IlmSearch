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

def _extract_text(data):
    """
    Helper to extract text from various JSON structures the API might return.
    It handles:
    1. Direct dictionary: {"text": "..."}
    2. List of hadiths: [{"text": "..."}]
    3. Nested 'hadiths' key: {"hadiths": [{"text": "..."}]}
    """
    target = data
    
    # Case A: Data is a list (e.g., [ {hadith...} ]) -> take the first one
    if isinstance(data, list):
        if not data: return ""
        target = data[0]
        
    # Case B: Data has a 'hadiths' key (wrapper) -> take the first item inside
    elif isinstance(data, dict) and "hadiths" in data:
        hadiths_list = data["hadiths"]
        if isinstance(hadiths_list, list) and hadiths_list:
            target = hadiths_list[0]

    # Now we have the single hadith object, look for the text field
    # Common keys are 'text', 'hadith', or 'body'
    return target.get("text") or target.get("hadith") or target.get("body") or ""

def _extract_grade(data):
    """
    Helper to extract grade safely.
    """
    target = data
    if isinstance(data, list) and data:
        target = data[0]
    elif isinstance(data, dict) and "hadiths" in data:
        if data["hadiths"]:
            target = data["hadiths"][0]
            
    return target.get("grades", []) or target.get("grade", [])

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
                return {"error": f"Hadith not found (API Status: {eng_resp.status_code})"}

            eng_data = eng_resp.json()
            ara_data = ara_resp.json() if ara_resp.status_code == 200 else {}
            
            # Use the helper to robustly find the text
            english_text = _extract_text(eng_data)
            arabic_text = _extract_text(ara_data)
            grades = _extract_grade(eng_data)

            return {
                "id": f"{clean_book}:{number}",
                "book": clean_book.capitalize(),
                "number": number,
                "english": english_text,
                "arabic": arabic_text,
                "grade": grades,
                "error": None
            }
        except Exception as e:
            return {"error": str(e)}