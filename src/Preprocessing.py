from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split



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

class PathTokenizer():
    """ 
		A simple class to tokenize the URL   
    """
    
    def __init__(self):
        self.stemmer = SnowballStemmer(language='french')
        self.stopwords = [unidecode(x) for x in stopwords.words('french')]
        self.special_words = ['htm', 'php', 'aspx', 'html']

    def _clean_text(self, text:str):
        """ 
        	Remove the symbols from the a url  
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
    
    def _remove_stopwords(self, tokens:list):
        if isinstance(tokens, list):
            return [t for t in tokens if t not in self.stopwords]
        else:
            raise TypeError("tokens must be a list")
            
    def _remove_single(self, tokens: list):
        """
			Remove single elements from list 
		"""
        if isinstance(tokens, list):
            return [t for t in tokens if len(t)>1]
        else:
            raise TypeError("tokens must be a list")
            
    def _remove_specials(self, tokens:list):
        if isinstance(tokens, list):
            return [t for t in tokens if t not in self.special_words]
        else:
            raise TypeError("tokens must be a list")
    
    def _remove_numbers(self, tokens:list):
        if isinstance(tokens, list):
            # return [x for x in text if not any(x1.isdigit() for x1 in x)]
            return [t for t in tokens if not t.isdigit()]
        else:
            raise TypeError("tokens must be a list") 
            
    def _stem_text(self, tokens:list):
        if isinstance(tokens, list):        
            return [self.stemmer.stem(token) for token in tokens]
        else:
            raise TypeError("tokens must be a list")        
        
    def _tokenize_text(self, text:str):
        return word_tokenize(text, language='french')
        
    def _join_words(self, text:list):
        """ Build a sentence from a list of words and separates them with a sapce"""
        return " ".join(text)
    
    def _split_words(self, text:str):
        return text.split(' ')
    
    def clean_df(self, df_column, funcs_list):
        "Apply multiple functions on a column of a dataframe"
        for func in funcs_list:
            df_column = df_column.apply(func)
        return df_column


def url_parse(url):
    parse_result = urlparse(url)
    result = [parse_result.scheme, parse_result.netloc, parse_result.path, parse_result.params, parse_result.query, parse_result.fragment]
    return result


def split_netloc(netloc:str):
    splited_netloc = netloc.rsplit('.', 2)
    if len(splited_netloc) == 2:
        splited_netloc.insert(0, "www")
    return splited_netloc

def filter_per_occurencies(df, n):
    """
        In each target list of the dataframe, delete all occurences of elements with a global count <= n.
    """
    df_copy = df.copy()
    count_labels = df_copy.explode('target')['target'].value_counts()
    stay = count_labels[count_labels>n].index
    df_copy['target'] = df_copy['target'].apply(lambda x: list(filter(lambda y: y in stay, x)))
    return df_copy


def parse_and_preprocess():
	"""
		Do all the pre-processing of the dataframe and return the final one.
	"""
	data = Path("../data/").glob("*.parquet") 
	data = list(data)
	df = pd.concat((pd.read_parquet(parquet, engine='fastparquet') for parquet in data))
	pd.set_option("max_colwidth", None)
	df = df.reset_index(drop=True)
	df_parsed = pd.concat([df, 
                       pd.DataFrame(list(map(url_parse, df.url)),
                    columns= ['scheme','netloc','path','params','quer','fragment'],
                   index=df.url.index) 
                       ], axis=1)
	df_parsed.drop(['params', 'quer', 'fragment'], axis=1, inplace=True)
	df_parsed_2 = pd.concat([df_parsed, 
						pd.DataFrame(list(map(split_netloc, df_parsed.netloc)),
						columns= ['sous_domaine','domaine','top_domaine'],
					index=df_parsed.netloc.index) 
						], axis=1)

	path_tokenizer = PathTokenizer()

	funcs = [path_tokenizer._clean_text, path_tokenizer._remove_numbers, path_tokenizer._remove_single, path_tokenizer._stem_text,
			path_tokenizer._lowercase_text, path_tokenizer._remove_specials, path_tokenizer._remove_stopwords ]

	df_parsed_2['tokens_path'] = path_tokenizer.clean_df(df_parsed_2.path, funcs)

	df_cleaned = df_parsed_2.drop(['url', 'path', 'scheme', 'netloc'], axis=1)

	mlb = MultiLabelBinarizer()
	targets_encoded = pd.DataFrame(mlb.fit_transform(df_cleaned.target),
					columns=mlb.classes_,
					index=df_cleaned.target.index)
	df_cleaned_2 = pd.concat([df_cleaned, targets_encoded], axis=1)
	df_cleaned_3 = df_cleaned_2.copy(deep=True).drop(columns=['target'])
	df_cleaned_3["tokens_path"] = df_cleaned_3.tokens_path.apply(path_tokenizer._join_words)
	return df_cleaned_3


def split_tfidf_and_onehotencoding(df):

	""" Splitting in the different sets  """
	y = df.iloc[:, 5:]
	x = df.iloc[:, : 5].drop(['sous_domaine'], axis=1)

	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

	categorical_features = ["day","domaine","top_domaine"]

	""" Preparing TF-IDF  """

	tfidf = TfidfVectorizer(min_df=0)
	vectorizer = TfidfVectorizer(min_df=0.00009, smooth_idf=True, norm="l2", tokenizer = lambda x: x.split(" "), sublinear_tf=False, ngram_range=(1,1))

	""" One Hot Encoding  """
	transformer = ColumnTransformer([ ('categorical', OneHotEncoder(sparse=False, handle_unknown = "ignore"), categorical_features),
								("vectorizer", vectorizer, "tokens_path"),
							], remainder="passthrough")

	X_train_multilabel = transformer.fit_transform(X_train)
	X_test_multilabel = transformer.transform(X_test)
	y_train_multilabel = y_train 
	y_test_multilabel = y_test 
	return X_train_multilabel, X_test_multilabel, y_train_multilabel, y_test_multilabel