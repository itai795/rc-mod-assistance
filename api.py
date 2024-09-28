from fastapi import FastAPI
from pydantic import BaseModel
from answer import answer_a_question

app = FastAPI()


class Question(BaseModel):
    user_question: str


@app.get("/ask_model/")
async def ask_model(question: Question):
    answer_data = answer_a_question(user_question=question.user_question)
    return answer_data.model_answer
