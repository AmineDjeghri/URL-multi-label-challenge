from preprocessing import Preprocess
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import spacy
from urllib.parse import urlparse
import re
import regex
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
# nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
from unidecode import unidecode
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
from sklearn.compose import ColumnTransformer
import fastparquet

def main():
    print("he")
    parquet_data_path = "../data/"
    preprocess = Preprocess()

    df = preprocess.load_dataframe("df.csv")
    print(df.columns)
    df = preprocess.preprocess_dataframe(df)

    print(df.columns)

if __name__ == '__main__':
    main()
