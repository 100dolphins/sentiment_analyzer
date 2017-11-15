from pymongo import MongoClient
import nltk

client = MongoClient()

db = client.test

collection = db.german_reviews

print('Building Training Set...')
negative_reviews = []
positive_reviews = []

for doc in collection.find({"stars": 1}):
    negative_reviews.append((doc['text'], 'negative'))


for doc in collection.find({"stars": 5}, limit=3026):
    positive_reviews.append((doc['text'], 'positive'))

reviews = []

for (words, sentiment) in positive_reviews + negative_reviews:
    words_filtered = [e.lower().strip('":,.!?();') for e in words.split() if len(e) >= 3]
    reviews.append((words_filtered, sentiment))


def get_words_in_reviews(reviews):
    all_words = []
    for (words, sentiment) in reviews:
        all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


word_features = get_word_features(get_words_in_reviews(reviews))


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


training_set = nltk.classify.apply_features(extract_features, reviews)

print('Training Naive Bayes Classifier...')
classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier.show_most_informative_features(20)

two_star_reviews = []
four_star_reviews = []

for doc in collection.find({"stars": 2}):
    two_star_reviews.append((doc['text'], 'negative'))


for doc in collection.find({"stars": 4}, limit=3026):
    four_star_reviews.append((doc['text'], 'positive'))

negative_test_reviews = []
positive_test_reviews = []

for (words, sentiment) in two_star_reviews:
    words_filtered = [e.lower().strip('":,.!?();') for e in words.split() if len(e) >= 3]
    negative_test_reviews.append((words_filtered, sentiment))

negative_test_set = nltk.classify.apply_features(extract_features, negative_test_reviews)

print('Accuracy for negative Reviews:')
print(nltk.classify.accuracy(classifier, negative_test_set))

for (words, sentiment) in four_star_reviews:
    words_filtered = [e.lower().strip('":,.!?();') for e in words.split() if len(e) >= 3]
    positive_test_reviews.append((words_filtered, sentiment))

word_features = get_word_features(get_words_in_reviews(positive_test_reviews))

positive_test_set = nltk.classify.apply_features(extract_features, positive_test_reviews)

print('Accuracy for positive Reviews:')
print(nltk.classify.accuracy(classifier, positive_test_set))
