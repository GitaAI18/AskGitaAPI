from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from sentence_transformers import SentenceTransformer, util

# Load model and data
model = SentenceTransformer("all-MiniLM-L6-v2")
with open("combined_questions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

app = FastAPI()

# CORS for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

@app.post("/ask")
async def ask_gita(query: Query):
    user_input = query.query
    query_embedding = model.encode(user_input, convert_to_tensor=True)

    best_score = 0
    best_entry = None
    for item in data:
        question = item.get("question_en", "")
        question_embedding = model.encode(question, convert_to_tensor=True)
        score = util.cos_sim(query_embedding, question_embedding).item()
        if score > best_score:
            best_score = score
            best_entry = item

    if best_score > 0.6:
        return {
            "question": best_entry["question_en"],
            "answer": best_entry["answer_en"],
            "shlokas": best_entry.get("shlokas", [])
        }
    else:
        return {"question": user_input, "answer": "🙏 Sorry, no suitable answer found.", "shlokas": []}
