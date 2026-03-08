import json
import os
import random
import requests
import asyncio
from google import genai
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Setup GenAI client for Evaluation
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
EVAL_MODEL = "gemini-3.1-flash-lite-preview"

# API Endpoint
ENDPOINT_URL = "http://localhost:8000/api1/process_question"
QUESTIONS_FILE = "data-example/questions.jsonl"

class EvalResult(BaseModel):
    is_correct: bool
    reason: str

async def evaluate_answer(question: str, expected_answer: str, actual_answer: str) -> EvalResult:
    prompt = f"""
    You are an objective evaluator.
    Assess if the 'Actual Answer' provided by an AI assistant correctly addresses the 'Question' 
    and contains the core facts from the 'Expected Answer'.
    
    Question: {question}
    Expected Answer: {expected_answer}
    Actual Answer: {actual_answer}
    
    Is the 'Actual Answer' correct based on the 'Expected Answer'?
    Provide a JSON response with:
    {{"is_correct": boolean, "reason": "short string explaining why it's correct or incorrect"}}
    """
    
    response = client.models.generate_content(
        model=EVAL_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json"
        )
    )
    
    try:
        data = json.loads(response.text)
        return EvalResult(is_correct=data.get('is_correct', False), reason=data.get('reason', str(data)))
    except Exception as e:
        return EvalResult(is_correct=False, reason=f"Parse Error: {e}")

def run_query(question: str):
    payload = {
        "content": question,
        "is_webhook": False,
        "dialog_id": "eval_test_01",
        "speaker_id": "test_speaker"
    }
    try:
        res = requests.post(ENDPOINT_URL, json=payload, timeout=30)
        res.raise_for_status()
        return res.json().get("answer", "")
    except Exception as e:
        print(f"Błąd HTTP dla pytania: {question} -> {e}")
        return ""

async def main():
    print("="*50)
    print("=== TEST RAG (LLM Evaluation) ===")
    print("="*50)
    
    # Check if API is running
    try:
        requests.get("http://localhost:8000/api1/health", timeout=5)
    except:
        print("BŁĄD: Serwer na localhost:8000 (uvicorn) NIE JEST URUCHOMIONY!")
        print("Uruchom najpierw: Start-Process -NoNewWindow C:\\Users\\pawel\\Desktop\\TAI\\server\\venv\\Scripts\\uvicorn.exe -ArgumentList \"adk_src.api:app --host 0.0.0.0 --port 8000\" (lub przez skrypt)")
        return

    # Load questions
    data = []
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # We will test 10 random questions
    test_sample = random.sample(data, min(10, len(data)))
    
    correct_count = 0
    total_count = len(test_sample)
    
    print(f"Wylosowano {total_count} pytań z {len(data)}. Rozpoczynamy test...\n")
    
    for idx, item in enumerate(test_sample):
        q = item['question']
        expected = item['answer']
        
        print(f"\n[{idx+1}/{total_count}] Pytanie: {q}")
        print(f"    Expected: {expected}")
        
        actual = run_query(q)
        print(f"    Actual:   {actual}")
        
        if not actual:
            print("    [!] Brak odpowiedzi z API.")
            continue
            
        eval_result = await evaluate_answer(q, expected, actual)
        
        print(f"    [EVAL] is_correct: {eval_result.is_correct}")
        print(f"           reason: {eval_result.reason}")
        
        if eval_result.is_correct:
            correct_count += 1
            
    print("\n" + "="*50)
    percent = (correct_count / total_count) * 100
    print(f"WYNIK KOŃCOWY: {correct_count}/{total_count} ({percent:.1f}%) poprawnych.")
    print("="*50)
    
if __name__ == "__main__":
    asyncio.run(main())
