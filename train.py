import pandas as pd
import numpy as np
from tqdm import tqdm
from model_util.bert_model import get_text_vector

df = pd.read_excel('./dataset/train_dataset.xlsx')

questions = df['QUESTION'].tolist()
answers = df['ANSWER'].tolist()


def create_and_save_vectors():
    question_vectors = []
    question_and_answer_vector = []

    for index, row in tqdm(df.iterrows()):
        question = row[0]
        answer = row[1]
        if question != None and answer != None:
            question_vectors.append(np.nan_to_num(np.array(get_text_vector(str(question))))[0])
            question_and_answer_vector.append(np.nan_to_num(np.array(get_text_vector(str(question) + str(answer))))[0])

    np.save('./dataset/question_vectors.npy', np.array(question_vectors, dtype=np.float64))
    np.save('./dataset/question_and_answer_vectors.npy', np.array(question_vectors, dtype=np.float64))
