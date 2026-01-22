import httpx
import asyncio

# Supported hadith books
SUPPORTED_BOOKS = {
    "bukhari", "muslim", "nawawi", "abudawud", "ibnmajah", "tirmidhi", "nasai"
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

    return target.get("text") or target.get("hadith") or target.get("body") or ""

def _extract_grade(data):

    target = data
    if isinstance(data, list) and data:
        target = data[0]
    elif isinstance(data, dict) and "hadiths" in data:
        if data["hadiths"]:
            target = data["hadiths"][0]
            
    return target.get("grades", []) or target.get("grade", [])

async def fetch_hadith(book: str, number: int):

    clean_book = book.lower().strip()
    
    if clean_book not in SUPPORTED_BOOKS:
        return {"error": f"Book '{book}' not found. Available: {sorted(SUPPORTED_BOOKS)}"}

    editions = {"eng": f"eng-{clean_book}", "ara": f"ara-{clean_book}"}
    base_url = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions"
    
    eng_url = f"{base_url}/{editions['eng']}/{number}.json"
    ara_url = f"{base_url}/{editions['ara']}/{number}.json"

    print(f"Fetching Hadith from URLs: {eng_url} and {ara_url}")
    
    async with httpx.AsyncClient() as client:
        try:
            eng_resp, ara_resp = await asyncio.gather(
                client.get(eng_url),
                client.get(ara_url)
            )

            if eng_resp.status_code != 200:
                return {"error": f"Hadith not found (API Status: {eng_resp.status_code})"}

            eng_data = eng_resp.json()
            ara_data = ara_resp.json() if ara_resp.status_code == 200 else {}
            
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

async def get_hadiths_by_refs(refs: list):
    """
    Each dict should have a 'ref' key like 'bukhari:1'.
    """
    async def dummy_error(error_msg):
        return {"error": error_msg}
    
    tasks = []
    for item in refs:
        ref = item.get("ref")
        if not ref:
            tasks.append(dummy_error("Missing ref"))
            continue
        try:
            book, num_str = ref.split(":", 1)
            number = int(num_str)
            tasks.append(fetch_hadith(book, number))
        except ValueError:
            tasks.append(dummy_error(f"Invalid ref format: {ref}"))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append({"error": str(result)})
        else:
            processed_results.append(result)
    return processed_results