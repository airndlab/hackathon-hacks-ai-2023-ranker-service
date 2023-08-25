import os
from model_util.bert_model import find_similarity_documents
import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/find_similarity")
async def say_hello(name: str):
    similarities = []
    for similarity in find_similarity_documents(name):
        if float(similarity['weight']) >= 0.5:
            similarities.append(similarity)

    return find_similarity_documents(name)
    

find_similarity_documents('санкт-петербург')

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=int(8085))