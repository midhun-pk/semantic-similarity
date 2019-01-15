import numpy as np
import math
import re

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

a = 100

class CorrelationBase:

	THRESHOLD = 0.70
	WORD_LENGTH_LIMIT = 2
	USE_PCA = True
	METHOD = 'TF-IDF' # 'SIF' # 
	COSINE_SIMILARITY_THRESHOLD = 0.75

	def __init__(self, word_vectors, path):
		self._path = path
		self._documents = {}
		self._word_vectors = word_vectors

	def get_path(self):
		return self.__path

	def is_empty(self, text = None):
		if text is not None: return len(text) == 0
		return len(self) == 0

	def remove_spaces(self, text):
		text = re.sub(r'[\t\n\r\s]+', ' ', text)
		return text.strip()

	def get_documents(self, token = ''):
		if token:
			return self._documents[token]
		return self._documents
	
	def add_document_frequency(self, id, token):
		'''
		@param self: Sentence
		@param id: String
		@param token: String
		Create a dict of tokens and their corresponding set of sentence ids.
		'''
		if token not in self._documents:
			self._documents[token] = set()
		self._documents[token].add(id)

	def get_frequency(self, token):
		pass

	def get_items(self):
		pass

	def write(self, a, b, similarity, writer):
		pass


	def find_term_frequency(self, token_frequency, total_tokens):
		return token_frequency / total_tokens
	
	def find_inverse_document_frequency(self, document_frequency, total_documents):
		return math.log(total_documents/document_frequency, 10) or 1
	
	def get_jaccard_similarity(self, a, b):
		a = set(a)
		b = set(b)
		sim = len(a.intersection(b)) / len(a.union(b))
		return sim

	def get_vectors(self):
		sentence_vectors = []
		word_vector_shape = np.array(list(map(float, list(self._word_vectors.values())[0]))).shape
		datas = self.get_items()
		for key, item in datas.items():
			sentence_vector = np.zeros(word_vector_shape)
			tokens = item.get_tokens()
			added_tokens = []
			for word in tokens:
				# Amplification factor x
				if word not in added_tokens:
					added_tokens.append(word)
					if CorrelationBase.METHOD == 'TF-IDF':
						term_frequency = self.find_term_frequency(item.get_token_frequency(word), len(tokens))
						inverse_document_frequency = self.find_inverse_document_frequency(len(self.get_documents(word)), len(datas))
						x = term_frequency * inverse_document_frequency
					else:
						a = 100
						x = a / (a + self.get_frequency(word))
					word_vector = np.array(list(map(float, self._word_vectors[word])))
					sentence_vector = np.add(sentence_vector, np.multiply(word_vector, x))
			if len(tokens) > 0:
				sentence_vector = np.divide(sentence_vector, len(tokens),)
			sentence_vectors.append(sentence_vector.tolist())
		sentence_vectors_np = np.array(sentence_vectors)
		if CorrelationBase.USE_PCA:
			svd = TruncatedSVD(n_components=1, n_iter=10, random_state=0)
			svd.fit(sentence_vectors_np)
			pc = svd.components_
			sentence_vectors_np = np.subtract(sentence_vectors_np, sentence_vectors_np.dot(pc.transpose()).dot(pc))
		return sentence_vectors_np

	def get_cosine_similarity(self, r_vectors, t_vectors):
		norm = np.linalg.norm(r_vectors, axis=1)[:, np.newaxis]
		normed_vectors = r_vectors/ norm
		vectors_transpose = np.transpose(normed_vectors)
		similarity = np.matmul(normed_vectors, vectors_transpose)
		np.fill_diagonal(similarity, 0)
		max_indices = np.argmax(similarity, axis = 1)
		return similarity, max_indices

	def find_correlation(self):
		vectors = self.get_vectors()
		similarity = self.get_cosine_similarity(vectors, vectors)
		return similarity

	def normalize(self, sim):
		sim = ( sim + 1 ) / 2
		sim = round(sim, 2)
		return sim