# LEMMATIZING
#A very similar operation to stemming is called lemmatizing. The major difference between these is,
# as you saw earlier, stemming can often create non-existent words, whereas lemmas are actual words.



import nltk
from nltk.stem import WordNetLemmatizer
# Because nltk data was downloaded in the D drive.
nltk.data.path.append('/media/dhanush/DATA/PyCharm_Projects_Ddrive/nltk_data')
lemmatizer = WordNetLemmatizer()


# Here, we've got a bunch of examples of the lemma for the words that we use. The only major thing to note is that
# lemmatize takes a part of speech parameter, "pos." If not supplied, the default is "noun."
# This means that an attempt will be made to find the closest noun, which can create trouble for you.
# Keep this in mind if you use lemmatizing!


print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better"))
print(lemmatizer.lemmatize("better", pos="a")) # pos = 'a' indicates adjective
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("ran",'v')) # pos = 'v' indicates verb