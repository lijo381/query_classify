#encoding='utf8'
import numpy as np
import pandas as pd
import spacy
import os
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
nlp = spacy.load("en")  #loading the spacy language model


curr_path=os.getcwd()
class TrainModel(object):
    def __init__(self, file_path):
        df = pd.read_table(file_path, sep=',,,', names=['q', 'category'],engine='python') #reading as data frame
        self.question = df.q
        self.cat = df.category
        self.features=[]
        self.Y=None
        self.X=None
        self.vocab=[]

    #function for extracting parts of speech based model using spacy for training
    def feature_generator(self,sent):
         token_list = ''
         ques=unicode(''.join([i if ord(i) < 128 else ' ' for i in sent]))
         doc=nlp(ques)
         #function giving different parts of speech
         for token in doc:
            dummy_list= token.pos_.encode('utf-8')+','+token.tag_.encode('utf-8')+','+token.dep_.encode('utf-8')+',' #POS: The simple part-of-speech tag.Tag: The detailed part-of-speech tag.Dep: Syntactic dependency, i.e. the relation between tokens
            token_list+=dummy_list
         self.features.append(token_list)

    #encoding the labels into one hot encoded vectors
    def labelIndexing(self):
        labels_cleaned=[label.strip() for label in self.cat]
        unique_labels=sorted(list(set(labels_cleaned))) #cleaning and sorting labels

        self.Y=np.asarray([unique_labels.index(l) for l in labels_cleaned]) #generating an index

    #function for generating word to vec using Tfid vectorization
    def feature_extractor(self):
        self.labelIndexing()
        self.question.apply(self.feature_generator)
        vectorized = TfidfVectorizer(lowercase=False)
        vector=vectorized.fit_transform(self.features).toarray()
        joblib.dump(vectorized, os.path.join(curr_path,'models/tfidf_vect.pkl'))
        vocab_all=vectorized.vocabulary_.keys()
        self.vocab=[v.encode('utf-8') for v in vocab_all]

        self.X=vector

    #train using random forest algorithm
    def train(self):
        self.feature_extractor()
        print ("Extracted features")
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.2)

        print("\n\n Train Statistics")
        print("Train",X_train[0].shape,Y_train.shape)
        print("Test",X_test.shape,Y_test.shape)
        model=RandomForestClassifier(n_estimators = 1000)
        model.fit(X_train, Y_train)
        print("Model trained...")
        joblib.dump(model, os.path.join(curr_path,'models/model.pkl'))
        print("Model saved %s" % os.path.join(curr_path,'models/model.pkl'))


