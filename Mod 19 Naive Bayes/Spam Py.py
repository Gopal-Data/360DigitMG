import pandas as pd #data manipulation 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB  
from sklearn.naive_bayes import GaussianNB  
from sklearn.naive_bayes import BernoulliNB
import re #cleaning the data

sms_raw = pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 19\\sms_raw_NB.csv",encoding = "ISO-8859-1")
sms_raw.shape
#Cleaning data
stop_words = []
with open("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 19\\stopwords_en.txt") as f:stop_words = f.read() #loading the stop words
stop_words = stop_words.split("\n") #breaking the single string to list

#Function tp clean the data
def cleaningdata (i):
    i= re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w= []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return(" ".join(w))
    
#Applying the custome function to email data text column
sms_raw["text"]= sms_raw["text"].apply(cleaningdata)

#Removing the Empty row
sms_raw = sms_raw.loc[sms_raw.text != " ",:]
 
#creating the predictors amd target
predictors = sms_raw.iloc[:,1]
target = sms_raw.iloc[:,0]

#Splitting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(predictors, target, test_size = 0.2)

#Creating a matrix of token counts for the entire text document
def split_if_words(i):
    return [word for word in i.split(" ")]

#Convert sms text into word count matrix (bag of words)
sms_count = CountVectorizer(analyzer = split_if_words).fit(sms_raw["text"])

#Applying the count matrix on entire sms data 
sms_matrix = sms_count.transform(sms_raw["text"])
sms_matrix.shape

#For training data
train_matrix = sms_count.transform(x_train)
train_matrix.shape

#For test data
test_matrix = sms_count.transform(x_test)
test_matrix.shape

#TFIDF transformation on  word count matrix
tfidf = TfidfTransformer().fit(sms_matrix)

#Applying Tfidf on train matrix data
train_tfidf = tfidf.transform(train_matrix)
train_tfidf.shape

#Applying Tfidf on test matrix data
test_tfidf = tfidf.transform(test_matrix)
test_tfidf.shape 

#Multinomial Naive Bayes model
M_NB = MultinomialNB()
M_NB.fit(train_tfidf,y_train)
M_NB_train = M_NB.predict(train_tfidf)
M_NB_train_Accu = np.mean(M_NB_train == y_train)
M_NB_train_Accu #.9718

pd.crosstab(M_NB_train, y_train)

M_NB_test = M_NB.predict(test_tfidf)
M_NB_test_Accu = np.mean(M_NB_test == y_test)
M_NB_test_Accu #0.9559

pd.crosstab(M_NB_test, y_test)

#Gaussiam naive bayes model
G_NB = GaussianNB()
G_NB.fit(train_tfidf.toarray(),y_train.values)
G_NB_train = G_NB.predict(train_tfidf.toarray())
G_NB_train_Accu = np.mean(G_NB_train == y_train)
G_NB_train_Accu #.8994

pd.crosstab(G_NB_train,y_train)

G_NB_test = G_NB.predict(test_tfidf.toarray())
accuracy_testgb_tfidf = np.mean(G_NB_test == y_test)
accuracy_testgb_tfidf #.8417

pd.crosstab(G_NB_test,y_test) 

#Gaussiam naive bayes model
G_NB = GaussianNB()
G_NB.fit(train_tfidf.toarray(),y_train.values)
G_NB_train = G_NB.predict(train_tfidf.toarray())
G_NB_train_Accu = np.mean(G_NB_train == y_train)
G_NB_train_Accu #.9060

pd.crosstab(G_NB_train,y_train) 

G_NB_test = G_NB.predict(test_tfidf.toarray())
accuracy_testgb_tfidf = np.mean(G_NB_test == y_test)
accuracy_testgb_tfidf #.8417

pd.crosstab(G_NB_test,y_test)

#Bernoulli Naive Bayes Model
B_NB = BernoulliNB()
B_NB.fit(train_tfidf.toarray(),y_train.values)
B_NB_train = B_NB.predict(train_tfidf.toarray())
B_NB_train_Accu = np.mean(B_NB_train == y_train)
B_NB_train_Accu #.9813

pd.crosstab(B_NB_train,y_train)

B_NB_test = B_NB.predict(test_tfidf.toarray())
accuracy_testgb_tfidf = np.mean(B_NB_test == y_test)
accuracy_testgb_tfidf #.9703

pd.crosstab(B_NB_test,y_test)