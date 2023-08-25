from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np

#Load AutoModel from huggingface model repository and dataframe with all data
tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")
df = pd.read_csv('./dataset/dataset.csv', sep=';', engine='python')
data_vectors = np.load('./dataset/vectors.npy')