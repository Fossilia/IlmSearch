from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, Query
from pydantic import BaseModel
from services.quran_service import get_quran_verses_by_refs
from services.openai_service import fetch_refs_from_openai
from services.hadith_service import fetch_hadith, get_hadiths_by_refs
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins (good for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuranResponse(BaseModel):
    reference: str
    surah_name: str | None = None
    verse_id: int | None = None
    arabic: str | None = None
    english: str | None = None
    error: str | None = None

class HadithResponse(BaseModel):
    id: str | None = None
    book: str | None = None
    number: int | None = None
    english: str | None = None
    arabic: str | None = None
    grade: list | None = None
    error: str | None = None

class SearchResult(BaseModel):
    quran: list[QuranResponse]
    hadith: list[HadithResponse]


@app.get("/search", response_model=SearchResult)
async def search_endpoint(
    query: str = Query(..., description="User query"),
    count: int = 5
):
    refs_dict = fetch_refs_from_openai(query, count)
    
    quran_refs = refs_dict.get("quran", [])
    hadith_refs = refs_dict.get("hadith", [])
    
    quran_results = get_quran_verses_by_refs(quran_refs)
    hadith_results = await get_hadiths_by_refs(hadith_refs)
    
    return {
        "quran": quran_results,
        "hadith": hadith_results
    }

@app.get("/hadith/{book}/{number}")
async def get_hadith(book: str, number: int):
    """
    Directly fetches a Hadith by book name and number.
    Example: GET /hadith/bukhari/1
    """
    result = await fetch_hadith(book, number)
    
    if result.get("error"):
        # Return a 404 if the hadith wasn't found
        raise HTTPException(status_code=404, detail=result["error"])
        
    return result