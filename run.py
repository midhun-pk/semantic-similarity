from SentenceCorrelation import Correlation
import json

fp =  open('vectors.json', 'r')
word_vectors = json.load(fp)
co = Correlation(word_vectors)
co.add_sentence('1', "Obama speaks to the media in Illinois")
co.add_sentence('2', "The President greets the press in Chicago")
co.add_sentence('3', "A girl is brushing her hair")
co.add_sentence('4', "A girl is styling her hair")
co.add_sentence('5', "A group of men play soccer on the beach")
co.add_sentence('6', "A group of boys are playing soccer on the beach")
co.add_sentence('7', "One woman is measuring another woman's ankle.")
co.add_sentence('8', "A woman measures another woman's ankle.")
co.add_sentence('9', "A man is sutting up a cucumber")
co.add_sentence('10', "A man is slicing up a cucumber")
co.add_sentence('11', "A man is playing a harp")
co.add_sentence('12', "A man is playing a keyboard")
scores, max_indices = co.find_correlation()

sentences = co.get_sentences_list()

for score in scores:
	score = (score + 1) / 2

for i in range(len(max_indices)):
	score = round((scores[i][max_indices[i]] + 1) / 2, 2)
	print(sentences[i].get_id() + ', ' + sentences[max_indices[i]].get_id() + ' - ' + str(score))
