export type QuranResult = {
  reference: string;
  surah_name: string | null;
  verse_id: number | null;
  arabic: string | null;
  english: string | null;
  error: string | null;
};

export type HadithResult = {
  id: string | null;
  book: string | null;
  number: number | null;
  english: string | null;
  arabic: string | null;
  grade: any[] | null;
  error: string | null;
};

export type SearchResult = {
  quran: QuranResult[];
  hadith: HadithResult[];
};

const API_URL = 'http://127.0.0.1:8000';

export const search = async (query: string, count: number): Promise<SearchResult> => {
  const url = `${API_URL}/search?query=${encodeURIComponent(query)}&count=${count}`;
  const response = await fetch(url);
  
  if (!response.ok) throw new Error(`Error: ${response.status}`);

  const data = await response.json();
  return data;
};