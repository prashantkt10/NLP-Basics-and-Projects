import numpy as np
import re
import pickle
import nltk
import sklearn
nltk.download('stopwords')

#import datasets
'''reviews = sklearn.datasets.load_files('txt_sentoken/')
X,y = reviews.data, reviews.target

# Lets store data as pickle files
with open('X.pickle', 'wb') as f:
    pickle.dump(X,f)
    
with open('y.pickle', 'wb') as f:
    pickle.dump(y,f)
'''    
    
# Unpickling the dataset
with open('X.pickle', 'rb') as f:
    X = pickle.load(f)
    
with open('y.pickle', 'rb') as f:
    y = pickle.load(f)
    

#Preprocess the data and Creating the corpus
corpus = []
for i in range(len(X)):
    review = re.sub(r'\W', ' ', str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+', ' ', review)
    review = re.sub(r'^[a-z]\s+', '', review)
    review = re.sub(r'\s+', ' ', review)
    corpus.append(review)
    
vectorizer = sklearn.feature_extraction.text.CountVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words=nltk.corpus.stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()