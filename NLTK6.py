# NAMED ENTITY RECOGNITION
# It comes built-in with NLTK
# The idea is to have the machine immediately be able to pull out "entities" like people, places,
# things, locations, monetary figures, and more.



import nltk
# Because nltk data was downloaded in the D drive.
nltk.data.path.append('/media/dhanush/DATA/PyCharm_Projects_Ddrive/nltk_data')
from nltk.corpus import state_union # State of the union addresses by various US presidents
from nltk.tokenize import PunktSentenceTokenizer
# The PunktSentenceTokenizer is an unsupervised ML sentence tokenizer. Pretrained, but can be retrained.

'''
NE Type and Examples
ORGANIZATION - Georgia-Pacific Corp., WHO
PERSON - Eddy Bonte, President Obama
LOCATION - Murray River, Mount Everest
DATE - June, 2008-06-29
TIME - two fifty a m, 1:30 p.m.
MONEY - 175 million Canadian Dollars, GBP 10.40
PERCENT - twenty pct, 18.75 %
FACILITY - Washington Monument, Stonehenge
GPE - South East Asia, Midlothian
# The NLTK POS tagger works with tokenized sentences, so you need to break your text into sentences
# and word tokens before you can POS tag.
'''

train_text = state_union.raw("2005-GWBush.txt") # we are grabbing the raw text
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text) # So we are training the tokenizer with the 2005 speech
tokenized = custom_sent_tokenizer.tokenize(sample_text) # Tokenizing 2006 speech

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            namedEnt = nltk.ne_chunk(tagged, binary = False)
            # Binary = True will classify chunks into NER or not ( 2 categories)
            # Binary = False will classify chunks into all the types given above


            print(namedEnt)
            namedEnt.draw()

    except Exception as e:
        print(str(e))

process_content()