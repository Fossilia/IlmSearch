from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, Query
from pydantic import BaseModel
from services.quran_service import search
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
    q: str = Query(..., description="User query"),
    k: int = 5
):
    return search(q, k)

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