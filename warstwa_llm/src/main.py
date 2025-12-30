import logging
import uvicorn
from .config import BASE_API_HOST, BASE_API_PORT
from .api import app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vector_search.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Główny punkt wejścia serwera API.

    Uruchamia serwer Uvicorn z aplikacją FastAPI.

    Hierarchia wywołań:
        warstwa_llm/src/main.py -> main() -> uvicorn.run(src.api.app)
    """
    logger.info(f"Uruchamianie serwera na {BASE_API_HOST}:{BASE_API_PORT}")
    uvicorn.run(app, host=BASE_API_HOST, port=BASE_API_PORT)

if __name__ == "__main__":
    main()