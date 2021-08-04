from scipy import sparse
from skmultilearn.embedding import OpenNetworkEmbedder
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
import joblib
import sys
from skmultilearn.embedding import EmbeddingClassifier
from sklearn.ensemble import RandomForestRegressor
from skmultilearn.adapt import MLkNN
from skmultilearn.ext import Meka
from skmultilearn.ext import download_meka

sys.modules['sklearn.externals.joblib'] = joblib
class LabelGraphModeling:
	def __init__(self, dimensions):
		graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
		openne_line_params = dict(batch_size=1000, order=3)
		embedder = OpenNetworkEmbedder(
			graph_builder,
			'LINE',
			dimension = 5*dimensions, # df_cleaned_3.iloc[:, 5:].values.shape[1]
			aggregation_function = 'add',
			normalize_weights=True,
			param_dict = openne_line_params
		)

		self.clf = EmbeddingClassifier(
			embedder,
			RandomForestRegressor(n_estimators=10),
			MLkNN(k=5)
		)

	def fit(self, X_train, y_train):
		self.clf.fit(X_train, y_train)

	def predict(self, X):
		return self.clf.predict(X_test_multilabel)


class MultiLabel_BinaryRelevance:
	def __init__(self):
		self.model = Meka(
			meka_classifier = "meka.classifiers.multilabel.BR", # Binary Relevance
			weka_classifier = "weka.classifiers.bayes.NaiveBayesMultinomial", # with Naive Bayes single-label classifier
			meka_classpath=download_meka(), # obtained via download_meka # for PC
			java_command='java'
		)


	def mat_transform(self, y):
		y_t = np.array(y, dtype=int)
		Y_sp = sparse.lil_matrix(y_t)
		return Y_sp

	def fit(self, X_train_multilabel, y_train):
		y_train_sp = self.mat_transform(y_train)
		self.model.fit(X_train_multilabel, y_train_sp)

	def predict(self, X_test, y_test):
		return self.predict(X_test)




class MultiLabel_PLST:
	def __init__(self):
		self.model = Meka(
		meka_classifier = "meka.classifiers.multilabel.PLST -size 3",
		meka_classpath=download_meka(), # obtained via download_meka # for PC
		java_command='java'  # path to java executable
		)



	def mat_transform(self, y):
		y_t = np.array(y, dtype=int)
		Y_sp = sparse.lil_matrix(y_t)
		return Y_sp

	def fit(self, X_train_multilabel, y_train):
		y_train_sp = self.mat_transform(y_train)
		self.model.fit(X_train_multilabel, y_train_sp)

	def predict(self, X_test, y_test):
		return self.predict(X_test)








