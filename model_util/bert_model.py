import pandas as pd
import numpy as np
import torch
import pymorphy3
import nltk
import re
from sklearn.metrics.pairwise import cosine_similarity
from model_util import model, tokenizer, df, data_vectors

__max_similarity_documents__ = 9

stop_words = ['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 
              'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему',
              'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 
              'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней',
              'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж',
              'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем',
              'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть',
              'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя',
              'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 
              'конечно', 'всю', 'между']

def prepare_text(text):
    word_tokenizer = nltk.WordPunctTokenizer()
    regex = re.compile(r'[А-Яа-яA-zёЁ-]+')
    text = " ".join(regex.findall(text)).lower()
    tokens = word_tokenizer.tokenize(text)
    morph = pymorphy3.MorphAnalyzer()
    # удаляем стоп-слова, а так же лемитизируем слова
    tokens = [morph.parse(word)[0].normal_form for word in tokens if (word not in stop_words and not word.isnumeric())]
    return ' '.join(tokens)

#Mean Pooling - Take attention mask into account for correct averaging
def __mean_pooling__(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_text_vector(text):
    text = prepare_text(text)
    
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=100, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    return __mean_pooling__(model_output, encoded_input['attention_mask'])

def __prepare_requet__(request):
    return np.nan_to_num(np.array(get_text_vector(request)))

def find_similarity_documents(request) -> []:
    results = []

    cosine_similarities = pd.Series(cosine_similarity(__prepare_requet__(request), data_vectors).flatten())

    for i,j in cosine_similarities.nlargest(__max_similarity_documents__).items():
        results.append({'weight':str(j), 'doc_id':int(df.id.iloc[i])
                        #, 'raw_text':df.text.iloc[i]*/
                        })

    return results