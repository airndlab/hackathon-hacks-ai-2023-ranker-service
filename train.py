import pandas as pd
import numpy as np
from tqdm import tqdm
from model_util.bert_model import get_text_vector

df = pd.read_csv('./dataset/dataset.csv', sep=';', engine='python')

texts = df['text'].tolist()

vectors = []
for text in tqdm(texts):
    vectors.append(np.nan_to_num(np.array(get_text_vector(text)))[0])

np.save('./dataset/vectors.npy', np.array(vectors, dtype=np.float64))
