import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('spam1.csv')

print df.shape   # (5572, 2)

print df.head()

print df.groupby('category').describe()

'''
         message                                                               
           count unique                                                top freq
category                                                                       
ham         4825   4516                             Sorry, I'll call later   30
spam         747    653  Please call our customer service representativ...    4
'''

df['spam1']=df['category'].apply(lambda x: 1 if x=='spam' else 0)
print df.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.message,df.spam1)
x_train_values = X_train.values[:25]
print x_train_values[:25] 
print type(X_train.values)    # <type 'numpy.ndarray'>
print y_train[:5]
'''
1407    0
2768    0
987     0
4177    0
598     0'''
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:2]

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)


emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)
model.predict(emails_count)

  # array([0, 1], dtype=int64)

X_test_count = v.transform(X_test)
print model.score(X_test_count, y_test)  # 0.9827709978463748

# Sklearn Pipeline

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

print clf.fit(X_train, y_train)
'''
Pipeline(memory=None,
     steps=[('vectorizer', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 1), preprocessor=None, stop_words=None,
        strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
        tokenizer=None, vocabulary=None)), ('nb', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))])'''


print clf.score(X_test,y_test)  # 0.9827709978463748

# Predicting the above two emails either spam or ham

print clf.predict(emails)  # array([0, 1], dtype=int64)




