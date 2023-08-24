import os

from fastapi import FastAPI

dir_path = os.getenv('DIR_PATH')
if dir_path is None:
    raise Exception('No DIR_PATH')

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/file/{filename}")
async def read_file(filename: str):
    with open(f'{dir_path}/{filename}') as file:
        content = file.read()
        return {"filename": filename, "content": content}
