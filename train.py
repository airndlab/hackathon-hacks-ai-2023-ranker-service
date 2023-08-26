import pickle

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from model_util.bert_model import get_text_vector


def create_and_save_vectors():
    df = pd.read_excel('./dataset/train_dataset.xlsx')
    question_vectors = []
    question_and_answer_vector = []

    for index, row in tqdm(df.iterrows()):
        question = row[0]
        answer = row[1]
        if question != None and answer != None:
            question_vectors.append(np.nan_to_num(np.array(get_text_vector(str(question))))[0])
            question_and_answer_vector.append(
                np.nan_to_num(np.array(get_text_vector(str(question) + ' ' + str(answer))))[0])

    np.save('./dataset/question_vectors.npy', np.array(question_vectors, dtype=np.float64))
    np.save('./dataset/question_and_answer_vectors.npy', np.array(question_vectors, dtype=np.float64))


def create_and_save_vectors_for_full_data():
    df_add_data = pd.read_csv('./dataset/content_v.2.csv', sep=';', encoding='cp1251')
    services = []
    services_vectors = []
    for index, row in tqdm(df_add_data.iterrows()):
        introtext = row['introtext']
        try:
            soup = BeautifulSoup(introtext, 'html.parser')
            text = ' '.join(soup.stripped_strings)
            services.append(text)
            services_vectors.append(np.nan_to_num(np.array(get_text_vector(text)))[0])
        except Exception as e:
            print("Ошибка при парсинге HTML:", e)

    all_data = {
        "services": services,
        "vectors": services_vectors,
    }
    # Сохранение списка на диск
    with open("./dataset/services.pkl", "wb") as f:
        pickle.dump(all_data, f)
