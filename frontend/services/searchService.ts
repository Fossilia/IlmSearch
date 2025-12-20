export type SearchResult = {
  reference: string;
  surah_name: string | null;
  verse_id: number | null;
  arabic: string | null;
  english: string | null;
  error: string | null;
};

const API_URL = 'http://127.0.0.1:8000';

export const searchQuran = async (query: string): Promise<SearchResult[]> => {
  const url = `${API_URL}/search?q=${encodeURIComponent(query)}&k=3`;
  const response = await fetch(url);
  
  if (!response.ok) throw new Error(`Error: ${response.status}`);

  const data = await response.json();
  return data;
};