import spacy
from sklearn.externals import joblib
import os

curr_pat=os.getcwd()
#model paths for tfid_vector and model
model_path=os.path.join(curr_pat,'models/model.pkl')
vector_p=os.path.join(curr_pat,'models/tfidf_vect.pkl')
nlp = spacy.load("en")
labels=['affirmation', 'unknown', 'what', 'when', 'who']  #available labels to classify into
class Predictor:
    def __init__(self,string):
        self.sentence=string
        self.features=[]
        self.X=None
        self.Y=None
    #Function for feature generation
    def feature_generator(self,sent):
         token_list = ''
         ques=unicode(''.join([i if ord(i) < 128 else ' ' for i in sent]))
         doc=nlp(ques)
         for token in doc:
            dummy_list= token.pos_.encode('utf-8')+','+token.tag_.encode('utf-8')+','+token.dep_.encode('utf-8')+','
            token_list+=dummy_list
         self.features.append(token_list)
         vectorized = joblib.load(vector_p)
         vector = vectorized.transform(self.features).toarray()
         self.X = vector
    #prediction function
    def pred(self):

        self.feature_generator(self.sentence) #generating the feature
        model=joblib.load(model_path)  #loading the random forest model
        result=model.predict(self.X)  #prediction
        return labels[result[0]]

if __name__=='__main__':

    string_to_predict="today is a sunny day"

    pred=Predictor(string_to_predict)
    prediction=pred.pred()
    print(string_to_predict,prediction)