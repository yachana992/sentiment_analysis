from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import re, string, random
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC
import nltk

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

dataset = positive_tweets + negative_tweets

label_p= [0 for x in positive_tweets]
label_n = [1 for x in negative_tweets]
label = label_p + label_n
data = {'data_set':dataset,'labels':label}
df = pd.DataFrame(data)
df = shuffle(df)
df.reset_index(inplace=True, drop=True)

y_train = list(df['labels'])

tweets = list(df['data_set'])

positive= df['labels'].value_counts()[0]
negative = df['labels'].value_counts()[1]

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
labels = ['positive', 'negative']
quantity = [positive, negative]
ax.bar(labels,quantity)
plt.xlabel('Quality of tweet')
plt.ylabel('Number of tweets')
plt.show()

print("The first 10 negative tweets")
print(df[df['labels'] == 1].head(10))

def remove_special_characters(text, remove_digits=True):
    text=re.sub(r'@[A-Za-z0-9]+','',text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text= re.sub(r'https?:\/\/\S+', '', text)
    return text

df['data_set']=df['data_set'].apply(remove_special_characters)

k=df['data_set'][0]

#tokenize
t=[]
for tweet in list(df['data_set']):
    tokened=word_tokenize(tweet)
    t.append(tokened)

#removing the stopwords
stop_words = set(stopwords.words('english'))
k=[]
for tweet in t:
    filtered_sentence = [w for w in tweet if not w in stop_words]
    k.append(filtered_sentence)

lemmatizer = WordNetLemmatizer()
h=[]
for tweet in k:
    x=[lemmatizer.lemmatize(word) for word in tweet]
    h.append(x)

df['normalized_tweet']=h
def check(a):
    if a==0:
        return 'positive tweet'
    else:
        return 'negative tweet'

m=[]
for tweet in h:
    r= [word for word in tweet if word !='user']
    m.append(r)

all_words = []

for tweet in m:
    for word in tweet:
        all_words.append(word.lower())

all_words = dict(nltk.FreqDist(all_words))

a = sorted(all_words.items(), key=lambda x: x[1],reverse=True) 
l=[]
for x,y in a:
    l.append(x)

word_features=l[:3000]

documents=[]
i=0
for tweet in m:
    x=(tweet,check(df['labels'][i]))
    i=i+1
    documents.append(x)

print("The tweet with corresponding label")
print(documents[0])

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(tweet), sentiment) for (tweet, sentiment) in documents]

print("The first feature set")
print(featuresets[0])

print("The length of the feature set")
print(len(featuresets))

training_set = featuresets[:7000]
testing_set = featuresets[7000:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Classifier accuracy percent on testing set:",(nltk.classify.accuracy(classifier, testing_set))*100,"%",sep='')
print("Classifier accuracy percent on training set:",(nltk.classify.accuracy(classifier, training_set))*100,"%",sep='')

print(classifier.show_most_informative_features(30))

MNB_classifier = SklearnClassifier(MultinomialNB())

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100,"%",sep="")

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100,"%",sep="")

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100,"%",sep="")

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100,"%",sep="")
