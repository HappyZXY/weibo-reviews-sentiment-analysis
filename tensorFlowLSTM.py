import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nltk
nltk.download("stopwords") # 下载停用词

from nltk.stem import SnowballStemmer # 提取词干
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re

MAX_WORDS = 100000 # 最大词汇量10万
MAX_SEQ_LENGTH = 30 # 最大序列长度30

def preprocessing(text, stem=False):
    stop_words = [line.strip() for line in  open('dataset/dic/stop.txt','r').readlines()]#停用词
    stemmer = SnowballStemmer('english')
    # 正则化表达式
    text_cleaning_re = r'@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+'
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token)) # 提取词干
            else:
                tokens.append(token) # 直接保存单词
    return ' '.join(tokens)


train_dataset, test_dataset = train_test_split(df, test_size = 0.2, random_state = 666, shuffle=True)

if __name__ == "__main__":
    pass