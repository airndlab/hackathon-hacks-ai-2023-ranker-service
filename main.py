import os
from model_util.bert_model import find_similarity_documents
from model_util import question_vectors, question_and_answer_vectors
import uvicorn
from fastapi import FastAPI
from train import create_and_save_vectors

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/find_similarity")
async def say_hello(question: str):
    similarities = []
    for similarity in find_similarity_documents(question, question_vectors):
        if float(similarity['weight']) >= 0.90:
            similarity['type'] = 'question'
            similarities.append(similarity)
    
    for similarity in find_similarity_documents(question, question_and_answer_vectors):
        if float(similarity['weight']) >= 0.70:
            similarity['type'] = 'question_and_answer'
            similarities.append(similarity)

    #TODO: убрать дубли, в приоритете оставить тип question

    return similarities


@app.get("/train")
async def train():
    # TODO: доделать асинхронный вызов функции построения векторов + добавить логирование !
    create_and_save_vectors()
    return "OK"

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=int(8085))