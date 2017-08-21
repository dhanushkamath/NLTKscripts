# STEMMING
# Stemming is basically removing ing and other things from the end of words. The meaning of the word remains the
# same. So we use stemming to get the main word from its variations (like past tense(ed), continuous(ing) etc)
# The stemmer used here is PorterStemmer which is very popular

import nltk
# Because nltk data was downloaded in the D drive.
nltk.data.path.append('/media/dhanush/DATA/PyCharm_Projects_Ddrive/nltk_data')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
example_words = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]

# for w in example_words:
#     print (ps.stem(w))

new_text = "It is very important to be pythonly while you are pythoning with python. All pythoners have pythoned poorly atleast once."
words = word_tokenize(new_text)

for w in words:
    print(ps.stem(w))

