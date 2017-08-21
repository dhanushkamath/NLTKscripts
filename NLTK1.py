# STOPWORDS
# Stopwords are words that do not help much or help very little in gaining the overall meaning of text such as
# the,i,me,who,myself,which,those etc. So we remove them


import nltk
# Because nltk data was downloaded in the D drive.
nltk.data.path.append('/media/dhanush/DATA/PyCharm_Projects_Ddrive/nltk_data')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = "This is an example showing off stop word filtration."
words = word_tokenize(example_sentence)
stop_words = set(stopwords.words("english"))

filtered_sentence = [w for w in words if w not in stop_words]
print(filtered_sentence)