import pandas as pd
import numpy as np
import os
import sys
import gensim

word_model = None
THREADS = os.cpu_count()
SIZE = 300
WINDOW = 5

# For string input s, return a list of the words it contains, in lowercase.
def preprocess(s : str):
    return [x.lower() for x in s.replace(" ", " ").split(" ")]

# For string input s, return a vector that represents the sum of the individual word vectors for the words that make up s.
def getVector(s):
    global word_model
    ourlist = []
    for i in preprocess(s):
        try:
            ourlist.append(word_model.wv[i])
        except:
            pass
    # print(len(ourlist))
    if (len(ourlist) == 0):
        return np.zeros(SIZE)
    else:
        return np.sum(ourlist, axis = 0)

# For list of strings input listoftext, create and return a model based on the Word2Vec algorithm.
def createModel(listoftext : list):
    global word_model
    data = []
    for i in listoftext:
        data += [preprocess(i)]
        # Try: vector_size=300, window=5, min_count=1
    word_model = gensim.models.Word2Vec(data, vector_size=SIZE, window=WINDOW, workers=THREADS)
    print("Total training time: {time:.3f}".format(time=word_model.total_train_time))
    return word_model

# Loads the Word2Vec model from file 'gigaword.model'
def loadModel():
    global word_model
    word_model = gensim.models.Word2Vec.load('dataset' + os.path.sep + 'gigaword.model')

# Creates and saves the Word2Vec model as file 'gigaword.model'
def CreateAndSaveModel():
    global word_model
    reviews = pd.read_csv('dataset' + os.path.sep + 'amazon.csv')
    word_model = createModel(reviews['Text'].tolist())
    word_model.save('dataset' + os.path.sep +'gigaword.model')

# Returns the Word2Vec model if it exists, otherwise it exits
def getWord2VecModel():
    global word_model
    if word_model != None:
        return word_model
    else:
        sys.exit(1)