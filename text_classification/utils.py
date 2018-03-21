import pandas as pd
import nltk
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
import random


def read_file(file):
    df = pd.read_csv(file, delimiter=',', encoding='utf8')
    tuples_list = [tuple(x) for x in df.values]
    return tuples_list

def get_data_dict(tuples):
    data_dict = {}
    for tuple in tuples:
        data_dict.setdefault(tuple[2], []).append(tuple[0])
    return data_dict

def remove_punctuation(text, tbl):
    # tbl = dict.fromkeys(i for i in range(sys.maxunicode)
    #                     if unicodedata.category(chr(i)).startswith('P'))
    return text.translate(tbl)

def get_tf_record(sentence, words):
    stemmer = LancasterStemmer()
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1

    return(np.array(bow))

def get_data_dict_from_csv(input_file):
    tuples = read_file(input_file)
    data_dict = get_data_dict(tuples)
    return data_dict


def get_words_n_docs(data, tbl):
    stemmer = LancasterStemmer()
    words = []
    # a list of tuples with words in the sentence and category name
    docs = []

    for each_category in data.keys():
        for each_sentence in data[each_category]:
            # remove any punctuation from the sentence
            #     this is very slow!!! :
            each_sentence = remove_punctuation(each_sentence, tbl)
            # print("sentence: " + each_sentence)
            # extract words from each sentence and append to the word list
            w = nltk.word_tokenize(each_sentence)
            # print("tokenized words: ", w)
            words.extend(w)
            docs.append((w, each_category))

    # stem and lower each word and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words]
    words = sorted(list(set(words)))
    return words, docs

def get_training(categories, docs, words):
    stemmer = LancasterStemmer()
    # create our training data
    training = []
    # create an empty array for our output
    output_empty = [0] * len(categories)

    for doc in docs:
        # initialize our bag of words(bow) for each document in the list
        bow = []
        # list of tokenized words for the pattern
        token_words = doc[0]
        # stem each word
        token_words = [stemmer.stem(word.lower()) for word in token_words]
        # create our bag of words array
        for w in words:
            bow.append(1) if w in token_words else bow.append(0)

        output_row = list(output_empty)
        output_row[categories.index(doc[1])] = 1

        # our training set will contain a the bag of words model and the output row that tells
        # which catefory that bow belongs to.
        training.append([bow, output_row])

    # shuffle our features and turn into np.array as tensorflow  takes in numpy array
    random.shuffle(training)
    training = np.array(training)
    return training

def get_tf_record(sentence, stemmer, words):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1

    return(np.array(bow))

def test_text(file, model, categories, words):
    stemmer = LancasterStemmer()
    with open(file) as f:
        lines = f.readlines()
    f.close()
    for line in lines:
        line = line.strip('\n')
        print (line)
        print(categories[np.argmax(model.predict([get_tf_record(line, stemmer, words)]))])
        print ("--------------------------")
