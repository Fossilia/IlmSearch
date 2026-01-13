from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, Query
from pydantic import BaseModel
from services.quran_service import get_quran_verses_by_refs
from services.openai_service import fetch_refs_from_openai
from services.hadith_service import fetch_hadith
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins (good for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchResponse(BaseModel):
    reference: str
    surah_name: str | None = None
    verse_id: int | None = None
    arabic: str | None = None
    english: str | None = None
    error: str | None = None


@app.get("/search", response_model=list[SearchResponse])
def search_endpoint(
    query: str = Query(..., description="User query"),
    count: int = 5
):
    refs = fetch_refs_from_openai(query, count)
    return get_quran_verses_by_refs(refs)

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