import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
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
import utils
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class Preprocess:
    def __init__(self):
        self.df = []

    def create_dataframe(self, parquet_data_path, preprocess=True):
        data = Path(parquet_data_path).glob("*.parquet")
        data = list(data)
        [print(parquet.name) for parquet in data]

        self.df = pd.concat((pd.read_parquet(parquet, engine='fastparquet') for parquet in data))
        self.df = self.df.reset_index(drop=True)

        if preprocess:
            self.df = self.preprocess_dataframe(self.df)

        return self.df

    def save_dataframe(self, dataframe, file_name="df.csv"):
        dataframe.to_csv(file_name, index=False)

    def load_dataframe(self, file_name):
        self.df = pd.read_csv(file_name)
        self.df = self.df.reset_index(drop=True)
        return self.df

    def parse_df_url(self, df, drop_columns=None):
        if drop_columns is None:
            drop_columns = ['params', 'quer', 'fragment']

        columns = ['scheme', 'netloc', 'path', 'params', 'quer', 'fragment']
        df_parsed = pd.concat([df,
                               pd.DataFrame(list(map(utils.url_parse, df.url)),
                                            columns=columns,
                                            index=df.url.index)
                               ], axis=1)
        df_parsed.drop(drop_columns, axis=1, inplace=True)
        return df_parsed

    def split_netloc(self, df_parsed):
        df_parsed_2 = pd.concat([df_parsed,
                                 pd.DataFrame(list(map(utils.split_netloc, df_parsed.netloc)),
                                              columns=['sous_domaine', 'domaine', 'top_domaine'],
                                              index=df_parsed.netloc.index)
                                 ], axis=1)
        return df_parsed_2

    def preprocess_dataframe(self, df, funcs=None):
        """default pipeline of preprocessing"""
        path_tokenizer = PathTokenizer()
        if funcs is None:
            funcs = [path_tokenizer._clean_text, path_tokenizer._remove_numbers, path_tokenizer._remove_single,
                     path_tokenizer._stem_text,
                     path_tokenizer._lowercase_text, path_tokenizer._remove_specials, path_tokenizer._remove_stopwords]
        df = self.parse_df_url(df)
        df = self.split_netloc(df)

        path_tokenizer = PathTokenizer()

        funcs = [path_tokenizer._clean_text, path_tokenizer._remove_numbers, path_tokenizer._remove_single,
                 path_tokenizer._stem_text,
                 path_tokenizer._lowercase_text, path_tokenizer._remove_specials, path_tokenizer._remove_stopwords]

        df_parsed_2 = df
        df_parsed_2['tokens_path'] = path_tokenizer.clean_df(df_parsed_2.path, funcs)

        df_cleaned = df_parsed_2.drop(['url', 'path', 'scheme', 'netloc'], axis=1)

        mlb = MultiLabelBinarizer()
        targets_encoded = pd.DataFrame(mlb.fit_transform(df_cleaned.target),
                                       columns=mlb.classes_,
                                       index=df_cleaned.target.index)
        df_cleaned_2 = pd.concat([df_cleaned, targets_encoded], axis=1)
        df_cleaned_2["tokens_path"] = df_cleaned_2.tokens_path.apply(path_tokenizer._join_words)
        return df_cleaned_2


class PathTokenizer():
    """ A simple class to tokenize the URL with a various combination of functions
        """

    def __init__(self):
        self.stemmer = SnowballStemmer(language='french')
        self.stopwords = [unidecode(x) for x in stopwords.words('french')]
        self.special_words = ['htm', 'php', 'aspx', 'html']

    def _clean_text(self, text: str):
        """
        remove the symbols from  a url
        """
        if isinstance(text, str):
            regex = '(\d+|[A-Z][a-z]*)|[+;,\s.!:\'/_%#&$@?~*]|-'
            t = list(filter(None, re.split(regex, text)))
            return t
        else:
            raise TypeError("text must be list")

    def _lowercase_text(self, tokens: list):
        if isinstance(tokens, list):
            return [t.lower() for t in tokens]
        else:
            raise TypeError("text must be list")

    def _remove_stopwords(self, tokens: list):
        if isinstance(tokens, list):
            return [t for t in tokens if t not in self.stopwords]
        else:
            raise TypeError("tokens must be a list")

    def _remove_single(self, tokens: list):
        "remove single elements from list "
        if isinstance(tokens, list):
            return [t for t in tokens if len(t) > 1]
        else:
            raise TypeError("tokens must be a list")

    def _remove_specials(self, tokens: list):
        if isinstance(tokens, list):
            return [t for t in tokens if t not in self.special_words]
        else:
            raise TypeError("tokens must be a list")

    def _remove_numbers(self, tokens: list):
        if isinstance(tokens, list):
            # return [x for x in text if not any(x1.isdigit() for x1 in x)]
            return [t for t in tokens if not t.isdigit()]
        else:
            raise TypeError("tokens must be a list")

    def _stem_text(self, tokens: list):
        if isinstance(tokens, list):
            return [self.stemmer.stem(token) for token in tokens]
        else:
            raise TypeError("tokens must be a list")

    def _join_words(self, text: list):
        """ build a sentence from a list of words and separates them with a sapce"""
        return " ".join(text)

    def _split_words(self, text: str):
        return text.split(' ')

    def clean_df(self, df_column, funcs_list):
        """Apply multiple functions on a column of a dataframe"""
        for func in funcs_list:
            df_column = df_column.apply(func)
        return df_column


def split_data(dataframe, test_size, categorical_features=None):

    # TO DO: add automated drop for the categories that are in categorical_features

    if categorical_features is None:
        categorical_features = ["day", "domaine", "top_domaine"]

    x = dataframe.iloc[:, : 6]
    y = dataframe.iloc[:, 6:]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    print(X_train.shape)
    print(X_test.shape)

    categorical_features = ["day", "domaine", "top_domaine"]

    tfidf = TfidfVectorizer(min_df=0)
    vectorizer = TfidfVectorizer(min_df=0.00009, smooth_idf=True, norm="l2", tokenizer=lambda x: x.split(" "),
                                 sublinear_tf=False, ngram_range=(1, 1))

    transformer = ColumnTransformer(
        [('categorical', OneHotEncoder(sparse=False, handle_unknown="ignore"), categorical_features),
         ("vectorizer", vectorizer, "tokens_path"),
         ], remainder="passthrough")

    target_train = X_train['target']
    target_test = X_test['target']

    X_train = X_train.drop(['sous_domaine', 'target'], axis=1)
    print(X_train.columns)
    X_train = transformer.fit_transform(X_train)

    X_test = X_test.drop(['sous_domaine', 'target'], axis=1)
    print(X_test.columns)
    X_test = transformer.transform(X_test)

    return X_train, X_test, y_train, y_test, target_train, target_test
