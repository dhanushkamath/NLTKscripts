# WORDNET
# WordNet is a lexical database for the English language, which was created by Princeton, and is part of the NLTK corpus.
# You can use WordNet alongside the NLTK module to find the meanings of words, synonyms, antonyms, and more

# A lemma is wordnet's version of an entry in a dictionary: A word in canonical form, with a single meaning.
# E.g., if you wanted to look up "banks" in the dictionary, the canonical form would be "bank" and there would be
# separate lemmas for the nouns meaning "financial institution" and "side of the river",
# a separate one for the verb "to bank (on)", etc.

# The term synset stands for "set of synonyms". A set of synonyms is a set of words with similar meaning, e.g. ship,
# skiff, canoe, kayak might all be synonyms for boat. In the nltk, a synset is in fact a set of lemmas with
# related meaning.

# edulix, stupidsid
# google usnews grad school rankings

import nltk
from nltk.corpus import wordnet
# Because nltk data was downloaded in the D drive.
nltk.data.path.append('/media/dhanush/DATA/PyCharm_Projects_Ddrive/nltk_data')

# SYNONYMS
syns = wordnet.synsets("good") # Gives the synsets ( a set of synonyms)
print(syns) # Prints the entire list of synsets
print(syns[0].lemmas())  # Prints the synonyms for the first synset
print(syns[0].lemmas()[0].name()) # To print only the word from the first list element

# Word Definitions
print(syns[0].definition()) # So it is seen that one synset has only one meaning. So all the lemmas that come inside are
                            # synonyms of the word with the same meaning, that can be used in the same context.

# Examples (context)
print(syns[0].examples())


synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        print("L:", l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")

print(w1.wup_similarity(w2)) # Comparing the similarity between 2 words

