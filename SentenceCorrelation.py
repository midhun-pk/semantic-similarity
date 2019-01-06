from CorrelationBase import CorrelationBase
from collections import OrderedDict

class Correlation(CorrelationBase):
	OUTPUT_CSV_FILE = 'Tensorboard.csv'
	class Sentence:
		ID_PREFIX = 'ID_'
		def __init__(self, id, description, tokens, token_frequency):
			self.__id = Correlation.Sentence.ID_PREFIX + id
			self.__description = description
			self.__tokens = tokens
			self.__token_frequency = token_frequency

		def get_id(self):
			return self.__id

		def get_description(self):
			return self.__description

		def get_tokens(self):
			return self.__tokens
		
		def get_token_frequency(self, token = ''):
			if token:
				return self.__token_frequency[token]
			return self.__token_frequency

	def __init__(self, path = None):
		super().__init__(path)
		self.__sentences = OrderedDict()
		self.__size = 0
		self.__frequency = {}
		self.__ignored_words = []

	def __len__(self):
		return self.__size
	
	def get_sentence(self, sentence_name):
		if sentence_name in self.__sentences:
			return self.__sentences[sentence_name]
		return None

	def get_sentences(self):
		return self.__sentences

	def get_items(self):
		return self.get_sentences()

	def get_frequency(self, token = ''):
		if token:
			return self.__frequency[token]
		return self.__frequency

	def get_ignored_words(self):
		return self.__ignored_words

	def increment_count(self, word):
		if word not in self.__frequency:
			self.__frequency[word] = 0
		self.__frequency[word] += 1

	def add_to_trash(self, word):
		if word not in self.__ignored_words:
			self.__ignored_words.append(word)

	def is_meaningful(self, word):
		if word in word_vectors and len(word) > CorrelationBase.WORD_LENGTH_LIMIT:
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

	def add_sentence(self, id, description):
		id = self.remove_extra_space(id)
		description = self.remove_extra_space(description)
		tokens, token_frequency = self.get_meaningful_tokens(id, description.lower())
		if not self.is_empty(id) and not self.is_empty(tokens):
			sentence = self.Sentence(id, description, tokens, token_frequency)
			self.__sentences[sentence.get_id()] = sentence
			self.__size += 1
			return sentence
		return None


