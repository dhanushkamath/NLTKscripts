# CORPUS
# Accessing the data from the CORPUS

import nltk
# Because nltk data was downloaded in the D drive.
nltk.data.path.append('/media/dhanush/DATA/PyCharm_Projects_Ddrive/nltk_data')
from nltk.tokenize import sent_tokenize,PunktSentenceTokenizer
from nltk.corpus import gutenberg

sample = gutenberg.raw("shakespeare-macbeth.txt")
tokens = sent_tokenize(sample)

print(tokens)







