from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import matplotlib.pyplot as plt
import re, string, random
import nltk.metrics
from wordcloud import WordCloud
from sklearn.metrics import f1_score
import collections
import numpy as np
from sklearn.metrics import classification_report
#Removing noises like punctuations, hyperlinks and stop words in the tweets
def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):

        #removing hyperlinks using regular expressions
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        #Removing @ using regular expression
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        #Removing the symbol,#
        token = re.sub(r'#','', token)
        #Removing the symbol RT for retweets
        token = re.sub(r'RT[\s]+','', token)
        token = token.replace("\'", "");
        token = token.replace("\"", "");

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        #Lemmatizing each word into its base form
        # Pos tagging each words
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        #Removes punctuations from the data and convert it into lowercase
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

if __name__ == "__main__":

    #Fetching the dataset
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]
    #print(tweet_tokens)
    #Stop words in english are words like the, and , is etc.
    stop_words = stopwords.words('english')

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    #Applying the remove noise method in postive dataset
    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    #Applying the remove noise method in negative dataset
    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    
    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    freq_dist_pos = FreqDist(all_pos_words)
    print(freq_dist_pos.most_common(10))
    
    wordcloud = WordCloud().generate(str(positive_cleaned_tokens_list))
    plt.figure(figsize = (15, 9))
    # Display the generated image:
    plt.imshow(wordcloud, interpolation= "bilinear")
    plt.axis("off")
    plt.show()

    wordcloud = WordCloud().generate(str(negative_cleaned_tokens_list))
    plt.figure(figsize = (15, 9))
    # Display the generated image:
    plt.imshow(wordcloud, interpolation= "bilinear")
    plt.axis("off")
    plt.show()

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Positive")
                         for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                         for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    #Shuffling the dataset 
    #random.shuffle(dataset)

    train_data = dataset[:7000]
    test_data = dataset[7000:]

    #Building the model 
    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy on test data is:", classify.accuracy(classifier, test_data))
    print("Accuracy on train data is:", classify.accuracy(classifier, train_data))

    print(classifier.show_most_informative_features(10))

    # The first elemnt of the tuple is the feature set and the second element is the label 
    ground_truth = [r[1] for r in test_data]

    preds = [classifier.classify(r[0]) for r in test_data]
    
    y_test = ground_truth
    y_pred = preds
    class_names = ['Negative', 'Positive']
    cm = nltk.ConfusionMatrix(y_test, y_pred)
    print(classification_report(ground_truth, preds))

    custom_tweet = "I do not enjoy this."

    custom_tokens = remove_noise(word_tokenize(custom_tweet))

    print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))
    
