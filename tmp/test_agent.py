import sys
import pathlib

# Upewnij się, że warstwa LLM jest importowalna
_project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src" / "warstwa_llm"))

from dotenv import load_dotenv
load_dotenv(dotenv_path=str(_project_root / ".env"), override=True)

import os
# upewniamy się, że klucz wpada do środowiska.
from adk_src.config import GEMINI_API_KEY
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

from agents.watus_agent.agent import popraw_nieczytelne_pytanie

tekst_testowy = "I-le ksionżka autor-ski i roz-dział w mono-grafjah o-publikowały pracowniki Wu-Ce-Ygrek w o-kres dwa ze-ro je-den o-siem myśl-nik dwa ze-ro dwa je-den py-taj-nik"

print(f"Dane wejściowe: {tekst_testowy}")
wynik = popraw_nieczytelne_pytanie(tekst_testowy)
print(f"Zwrócony wynik:")
print(wynik)
