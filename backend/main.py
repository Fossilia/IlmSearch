from fastapi import FastAPI, Query
from pydantic import BaseModel
from services.quran_service import search

app = FastAPI()

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
