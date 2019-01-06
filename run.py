from SentenceCorrelation import Correlation

co = Correlation()
co.add_sentence(1, "This is a cat")
co.add_sentence(2, "This is a kitten")
co.find_correlation()
