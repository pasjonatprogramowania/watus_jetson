"""
Watus Agent — główny agent ADK (Google Agent Development Kit).

Ten plik definiuje `root_agent`, który jest wymagany przez ADK Web Inspector
aby poprawnie zarejestrować agenta i umożliwić interakcję z nim w przeglądarce.

Agent korzysta z bazy wiedzy Qdrant (mem0) do odpowiadania na pytania o WAT.
Posiada również narzędzie do przeszukiwania internetu (DuckDuckGo).
"""

import os
import sys
import pathlib

# Upewnij się, że warstwa LLM jest importowalna
_project_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src" / "warstwa_llm"))

# Załaduj .env
from dotenv import load_dotenv
load_dotenv(dotenv_path=str(_project_root / ".env"), override=True)

from google.adk.agents import Agent

# Import konfiguracji z adk_src
from adk_src.config import (
    CURRENT_MODEL,
    DEFAULT_SYSTEM_PROMPR,
    GEMINI_API_KEY,
)

# Ustaw klucz API w środowisku (wymagany przez ADK)
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY


# ─── Narzędzia (Tools) ───────────────────────────────────────
def wyszukaj_w_internecie(zapytanie: str) -> dict:
    """Wyszukuje informacje w internecie za pomocą DuckDuckGo.
    Użyj tego narzędzia, gdy użytkownik pyta o aktualności, pogodę, 
    wydarzenia, lub cokolwiek co wymaga dostępu do internetu.
    
    Args:
        zapytanie: Tekst zapytania do wyszukania w internecie.
    
    Returns:
        dict: Wyniki wyszukiwania zawierające tytuły, opisy i linki.
    """
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(zapytanie, max_results=5))
        
        if not results:
            return {"status": "brak_wynikow", "message": "Nie znaleziono wyników."}
        
        formatted = []
        for r in results:
            formatted.append({
                "tytul": r.get("title", ""),
                "opis": r.get("body", ""),
                "link": r.get("href", "")
            })
        
        return {"status": "ok", "wyniki": formatted}
    except Exception as e:
        return {"status": "blad", "message": f"Błąd wyszukiwania: {str(e)}"}

# ─── Root Agent ───────────────────────────────────────────────
def opisz_zdjecie(sciezka_do_pliku: str) -> dict:
    """Analizuje zdjęcie i opisuje co na nim widać.
    Użyj tego narzędzia, gdy użytkownik prześle zdjęcie lub poda ścieżkę do pliku
    graficznego i chce wiedzieć co jest na zdjęciu.

    Args:
        sciezka_do_pliku: Ścieżka do pliku graficznego (jpg, png, webp, gif, bmp) 
                          lub URL do obrazka w internecie.

    Returns:
        dict: Opis tego co widać na zdjęciu.
    """
    try:
        from google import genai
        from google.genai import types
        import base64
        import mimetypes

        client = genai.Client()

        # Sprawdź czy to URL
        if sciezka_do_pliku.startswith(("http://", "https://")):
            import requests as req
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            try:
                resp = req.get(sciezka_do_pliku, timeout=15, headers=headers, allow_redirects=True)
                resp.raise_for_status()
            except req.exceptions.ConnectionError:
                return {"status": "blad", "message": f"Nie można połączyć się z serwerem. Sprawdź URL lub połączenie z internetem: {sciezka_do_pliku}"}
            except req.exceptions.HTTPError as he:
                return {"status": "blad", "message": f"Serwer zwrócił błąd HTTP {resp.status_code} dla: {sciezka_do_pliku}"}
            image_bytes = resp.content
            content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0]
        else:
            # Plik lokalny
            file_path = pathlib.Path(sciezka_do_pliku)
            if not file_path.exists():
                return {"status": "blad", "message": f"Plik nie istnieje: {sciezka_do_pliku}"}
            
            image_bytes = file_path.read_bytes()
            mime, _ = mimetypes.guess_type(str(file_path))
            content_type = mime or "image/jpeg"

        image_part = types.Part.from_bytes(data=image_bytes, mime_type=content_type)

        response = client.models.generate_content(
            model=CURRENT_MODEL,
            contents=[
                "Opisz szczegółowo co widzisz na tym zdjęciu. Odpowiedz po polsku.",
                image_part,
            ],
        )

        return {"status": "ok", "opis": response.text}

    except Exception as e:
        return {"status": "blad", "message": f"Błąd analizy zdjęcia: {str(e)}"}


def popraw_nieczytelne_pytanie(nieczytelny_tekst: str) -> str:
    """Narzędzie do poprawiania nieczytelnych wypowiedzi użytkownika, które zostały źle przetworzone przez system rozpoznawania mowy (błędy ASR).
    Używaj tego narzędzia TYLKO wtedy, gdy wypowiedź użytkownika jest zupełnie niezrozumiała, przypomina bełkot fonetyczny 
    (np. "I-le ksionżka autor-ski...") i nie potrafisz z niej samodzielnie wywnioskować poprawnej intencji. 
    NIE UŻYWAJ tego narzędzia do odpowiadania na sensowne i w miarę poprawne pytania.
    
    Args:
        nieczytelny_tekst: Tekst wypowiedzi użytkownika, który wydaje się być zlepkiem przypadkowych sylab lub błędem transkrypcji.
        
    Returns:
        dict: Sformułowane pytanie np. {"status": "ok", "wiadomosc": "Czy chodziło ci o: <ZREKONSTRUOWANE_ZAPYTANIE>?"}
    """
    try:
        from google import genai
        client = genai.Client()
        prompt = (
            "Twoim zadaniem jest wydobycie sensu z wypowiedzi użytkownika, która została źle przetworzona przez system "
            "rozpoznawania mowy (TTS/ASR) i brzmi nieczytelnie. Spróbuj zrekonstruować oryginale zapytanie użytkownika.\n\n"
            f"Wypowiedź użytkownika do rozszyfrowania: {nieczytelny_tekst}\n\n"
            "Zwróć TYLKO samo zrekonstruowane zapytanie, w poprawnej polszczyźnie, bez żadnych dodatkowych komentarzy. "
            "Nie odpowiadaj na nie, tylko zrekonstruuj tekst."
        )
        response = client.models.generate_content(
            model=CURRENT_MODEL,
            contents=prompt,
        )
        zrekonstruowane = response.text.strip()
        return {"status": "ok", "wiadomosc": f"Czy chodziło ci o: {zrekonstruowane}?"}
    except Exception as e:
        return {"status": "blad", "message": f"Wystąpił błąd podczas próby odczytania zapytania: {str(e)}. Powiedz użytkownikowi, że nie zrozumiałeś i poproś o powtórzenie."}


INSTRUCTION = DEFAULT_SYSTEM_PROMPR + """

Masz do dyspozycji narzędzia:
1. `wyszukaj_w_internecie` — użyj go, gdy użytkownik pyta o coś wymagające aktualnych informacji 
   z internetu (pogoda, wiadomości, wydarzenia, ceny, kursy walut itp.).
2. `opisz_zdjecie` — użyj go, gdy użytkownik prześle zdjęcie lub poda ścieżkę/URL do obrazka 
   i chce wiedzieć co jest na nim widoczne. Podaj ścieżkę do pliku lub URL jako argument.
3. `popraw_nieczytelne_pytanie` — użyj go ZANIM spróbujesz odpowiedzieć użytkownikowi, gdy jego wiadomość brzmi
   jak bełkot z systemu ASR (np. literowanie, dziwne łamanie słów, fonetyczny zapis bez sensu, "I-le ksionżka autor-ski"). 
   Gdy to narzędzie zwróci zrekonstruowany tekst (np. "Czy chodziło ci o: ...?"), TWOJĄ OSTATECZNĄ I JEDYNĄ ODPOWIEDZIĄ
   DLA UŻYTKOWNIKA musi być DOKŁADNIE TEN ZWRÓCONY TEKST. Absolutnie nie odpowiadaj na zrekonstruowane zapytanie,
   tylko zacytuj wynik z narzędzia, aby dopytać użytkownika, czy o to mu chodziło. 
"""

root_agent = Agent(
    name="watus_asystent",
    model=CURRENT_MODEL,
    description="Asystent WAT z dostępem do internetu i analizą zdjęć.",
    instruction=INSTRUCTION,
    tools=[wyszukaj_w_internecie, opisz_zdjecie, popraw_nieczytelne_pytanie],
)
