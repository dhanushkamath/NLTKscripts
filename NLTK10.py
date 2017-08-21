# TEXT CLASSIFIER
# Sentiment analysis using NAIVE BAYES
# Saving classifier
import nltk
import pickle
import random
# Because nltk data was downloaded in the D drive.
nltk.data.path.append('/media/dhanush/DATA/PyCharm_Projects_Ddrive/nltk_data')
from nltk.corpus import movie_reviews  # 1000 positive and 1000 negative movie reviews, and they are already labelled

# The use and working of the for loop is given a few liens below.
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffling the training data
random.shuffle(documents)

# Expanding the same code above from one liner to multiple lines for better understanding. BOTH ARE THE SAME.
# documents =[]
# for category in movie_reviews.categories():
#     for fileid in movie_reviews.fileids(category):
#         documents.append(list(movie_reviews.words(fileid)),category)


# print(documents[1])
#  As seen, It is a tuple with 2 elements. The first element is a list of words in THIS PARTICULAR REVIEW.
#  and the second element is either 'pos' or 'neg'
# So each element in the list is a tuple with a list of words from one review and the label being the second element.
# This is what the for loop above is used for.

# we take all the words from all the reviews

all_words = []
for w in movie_reviews.words(): # Takes all the words from all the movie reviews
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
# Creates a list of NLTK freq distribution. Each element in the list is a dictionary
# Key is The word and the value is number of occurrence of that word.


#print(all_words.most_common(15)) # Printing the first 15 most common words
#print(all_words["stupid"]) # To get the number of occurrence of the word stupid
#print(all_words.keys()) # it is seen that punctuations are also present

word_features = list(all_words.keys())[:3000] # Takes 3000 words as features
#print(word_features)


# Defining a function to find the features in the documents

def find_features(document):
    words = set(document) # Sets are lists with no duplicate entries. It creates a list of all unique words in this doc.
    # Just one iteration of every unique word.

    features = {}
    for w in word_features:
        features[w] = (w in words) # This is a boolean. So if the word is present it stores True in features else False

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev),category) for (rev,category) in documents]
# So it converts (entire review,label) to (word features,label) which can now be trained

training_set = featuresets[:1900]
testing_set = featuresets[1900:]


# NAIVE BAYES ALGORITHM is a very simple and basic algorithm
# posterior (likelihood) = prior occurrences x likelihood/ evidence

classifier = nltk.NaiveBayesClassifier.train(labeled_featuresets=training_set)
# labeled_featuresets: A list of classified featuresets,i.e., a list of tuples (featureset, label)
# From the documentation, it is observed that this accepts training data in the format [reviews,labels]
print("Naive Bayes Accuracy :", nltk.classify.accuracy(classifier, testing_set)*100)
classifier.show_most_informative_features(15)

save_classifier = open("naivebayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

# Now for loading
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
print("Naive Bayes Accuracy :", nltk.classify.accuracy(classifier, testing_set)*100)
classifier.show_most_informative_features(15)







