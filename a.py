import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
import sklearn
import codecs
import pymorphy2
import seaborn as sns
sns.set_style("darkgrid")
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords 
nltk.download('stopwords')

def list_to_str(arr):
    str_ = ''
    for rec in arr:
        str_+=rec
    return str_


def df_preprocess(text):    
    reg = re.compile('[^а-яА-яa-zA-Z0-9 ]') #
    text = text.lower().replace("ё", "е")
    text = text.replace("ъ", "ь")
    text = text.replace("й", "и")
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'сайт', text)
    text = re.sub('@[^\s]+', 'пользователь', text)
    text = reg.sub(' ', text)
    stopWords = set(stopwords.words('russian'))
    # Стемминг
    stemmer = SnowballStemmer("russian")
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stopWords])
    return text


RUSTORE_CSV_POS = r'result\api\rustore.ru\csv\revData_rustore.ru_pos.csv'
RUSTORE_CSV_NEG = r'result\api\rustore.ru\csv\revData_rustore.ru_neg.csv'

rustore_neg = pd.read_csv(RUSTORE_CSV_POS)
rustore_pos = pd.read_csv(RUSTORE_CSV_NEG)
rustore_neg['Text'] = rustore_neg['Text'].apply(df_preprocess)
rustore_pos['Text'] = rustore_pos['Text'].apply(df_preprocess)

SRAVNI_CSV_POS = r'result\api\sravni.ru\csv\revData_sravni.ru_pos.csv'
SRAVNI_CSV_NEG = r'result\api\sravni.ru\csv\revData_sravni.ru_neg.csv'

# sravni_neg = pd.read_csv(SRAVNI_CSV_POS)
# sravni_pos = pd.read_csv(SRAVNI_CSV_NEG)

# sravni_neg['Text'] = sravni_neg['Text'].apply(df_preprocess)
# sravni_pos['Text'] = sravni_pos['Text'].apply(df_preprocess)


# rustore_pos_texts = []
# for i in range(len(rustore_pos)):
#     arr = ""
#     for elem in rustore_pos['Text'][i]:
#         arr += str(elem)
#         arr += ' '
#     rustore_pos_texts.append(arr)


# rustore_neg_texts = []
# for i in range(len(rustore_neg)):
#     arr = ""
#     for elem in rustore_neg['Text'][i]:
#         arr += str(elem)
#         arr += ' '
#     rustore_neg_texts.append(arr)


# sravni_pos_texts = []
# for i in range(len(sravni_pos)):
#     arr = ""
#     for elem in sravni_pos['Text'][i]:
#         arr += str(elem)
#         arr += ' '
#     rustore_pos_texts.append(arr)


# sravni_neg_texts = []
# for i in range(len(sravni_neg)):
#     arr = ""
#     for elem in sravni_neg['Text'][i]:
#         arr += str(elem)
#         arr += ' '
#     rustore_neg_texts.append(arr)


vectorizer = TfidfVectorizer() 
vectors = vectorizer.fit_transform(rustore_pos['Text'])
print("n_samples: %d, n_features: %d" % vectors.shape)
