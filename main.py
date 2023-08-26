import uvicorn
from fastapi import FastAPI

from model_util import question_vectors, question_and_answer_vectors
from model_util.bert_model import find_similarity_documents
from train import create_and_save_vectors

app = FastAPI(
    title='Ranker API'
)


def not_duplicate(similarity: dict, similarities: list):
    return all(similarity['id'] != item['id'] for item in similarities)


@app.get("/find_similarity")
async def find_similarity(question: str):
    similarities = []
    for similarity in find_similarity_documents(question, question_vectors):
        if float(similarity['weight']) >= 0.90:
            similarity['type'] = 'question'
            similarities.append(similarity)

    for similarity in find_similarity_documents(question, question_and_answer_vectors):
        if float(similarity['weight']) >= 0.70:
            similarity['type'] = 'question_and_answer'
            if not_duplicate(similarity, similarities):
                similarities.append(similarity)

    return similarities


@app.get("/train")
async def train():
    # TODO: доделать асинхронный вызов функции построения векторов + добавить логирование !
    create_and_save_vectors()
    return "OK"


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=int(8085))
