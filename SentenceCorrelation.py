from CorrelationBase import CorrelationBase
from collections import OrderedDict
import re

class Correlation(CorrelationBase):
	OUTPUT_CSV_FILE = 'Tensorboard.csv'
	class Sentence:
		ID_PREFIX = ''
		def __init__(self, id, text, tokens, token_frequency):
			self._id = Correlation.Sentence.ID_PREFIX + str(id)
			self._text = text
			self._tokens = tokens
			self._token_frequency = token_frequency

		def get_id(self):
			return self._id

		def get_text(self):
			return self._text

		def get_tokens(self):
			return self._tokens
		
		def get_token_frequency(self, token = ''):
			if token:
				return self._token_frequency[token]
			return self._token_frequency

	def __init__(self, word_vectors, path = None):
		super().__init__(word_vectors, path)
		self._sentences = OrderedDict()
		self._size = 0
		self._frequency = {}
		self._ignored_words = []

	def __len__(self):
		return self._size
	
	def get_sentence(self, sentence_name):
		if sentence_name in self._sentences:
			return self._sentences[sentence_name]
		return None

	def get_sentences(self):
		return self._sentences

	def get_sentences_list(self):
		return list(self._sentences.values())

	def get_items(self):
		return self.get_sentences()

	def get_frequency(self, token = ''):
		if token:
			return self._frequency[token]
		return self._frequency

	def get_ignored_words(self):
		return self._ignored_words

	def increment_count(self, word):
		if word not in self._frequency:
			self._frequency[word] = 0
		self._frequency[word] += 1

	def add_to_trash(self, word):
		if word not in self._ignored_words:
			self._ignored_words.append(word)

	def is_meaningful(self, word):
		if word in self._word_vectors and len(word) > CorrelationBase.WORD_LENGTH_LIMIT:
			return True
		return False

	def tokenize(self, sentence):
		return re.split(r"[^a-zA-Z0-9]+", sentence)
	
	def get_meaningful_tokens(self, id, sentence):
		token_frequency = {}
		meaningful_tokens = []
		tokens = self.tokenize(sentence)
		#tokens = set(tokens)
		for token in tokens:
			if token and self.is_meaningful(token):
				if token not in token_frequency:
					token_frequency[token] = 0
				token_frequency[token] += 1
				meaningful_tokens.append(token)
				self.add_document_frequency(id, token)
				self.increment_count(token)
		return meaningful_tokens, token_frequency

	def add_sentence(self, id = '', description = ''):
		id = self.remove_spaces(id)
		description = self.remove_spaces(description)
		tokens, token_frequency = self.get_meaningful_tokens(id, description.lower())
		if not self.is_empty(id) and not self.is_empty(tokens):
			sentence = self.Sentence(id, description, tokens, token_frequency)
			self._sentences[sentence.get_id()] = sentence
			self._size += 1
			return sentence
		return None


