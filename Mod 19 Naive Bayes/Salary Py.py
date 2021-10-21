import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB  
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB  
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
 
#loading the dataset
salary_train = pd.read_csv("C:\\Users\\gopal\Documents\\360DigiTMG\\mod 19\\SalaryData_Train.csv")
salary_test = pd.read_csv("C:\\Users\\gopal\Documents\\360DigiTMG\\mod 19\\SalaryData_Test.csv")

#Selecting the variable with caterforical variable
Categorical_Variable = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

#Tycasting both train and test dataset to numberic with labelEncoder
for i in Categorical_Variable:
    salary_train[i]= LabelEncoder().fit_transform(salary_train[i])
    salary_test[i]=LabelEncoder().fit_transform(salary_test[i])
    
#Spilting the data into X and Y , Train and Test
x_train = salary_train[salary_train.columns [0:13]]
y_train = salary_train[salary_train.columns [13]]
x_test = salary_test[salary_test.columns [0:13]]
y_test = salary_test[salary_test.columns [13]]
 
# Multinomial Naive Bayes Model 
M_NB = MultinomialNB()
M_NB.fit(x_train,y_train) 
M_NB_Pred = M_NB.predict(x_train)
M_NB_Accu_Train = np.mean(M_NB_Pred == y_train)
M_NB_Accu_Train #.772 

#Confustion Matrix
pd.crosstab(M_NB_Pred, y_train)

#Test Data
M_NB_Pred_Test = M_NB.predict(x_test)
M_NB_Accu_Test = np.mean(M_NB_Pred_Test == y_test)
M_NB_Accu_Test #.774

#Confustion Matrix on test data
pd.crosstab(M_NB_Pred_Test, y_test)

#Gaussian model
G_NB = GaussianNB()
G_NB.fit(x_train, y_train)
G_NB_Pred = G_NB.predict(x_train)
G_NB_Accu_Train = np.mean(G_NB_Pred == y_train)
G_NB_Accu_Train #.795

pd.crosstab(G_NB_Pred,y_train)

#Test data
G_NB_Pred_Test = G_NB.predict(x_test)
G_NB_Accu_Test = np.mean(G_NB_Pred_Test == y_test)
G_NB_Accu_Test #0.79

pd.crosstab(G_NB_Pred_Test, y_test)


#Categorical model
C_NB = CategoricalNB()
C_NB.fit(x_train, y_train)
C_NB_Pred = C_NB.predict(x_train)
C_NB_Accu_Train = np.mean(G_NB_Pred == y_train)
C_NB_Accu_Train #0.795

pd.crosstab(C_NB_Pred,y_train)

#Test Data
C_NB_Pred_Test =C_NB.predict(x_test)
C_NB_Accu_Test = np.mean(C_NB_Pred_Test == y_test)
C_NB_Accu_Test #0.856 

pd.crosstab(C_NB_Pred_Test,y_test)

#Bernoulli model
B_NB = BernoulliNB()
B_NB.fit(x_train, y_train)
B_NB_Pred = B_NB.predict(x_train)
B_NB_Accu_Train = np.mean(B_NB_Pred == y_train)
B_NB_Accu_Train #0.725

pd.crosstab(B_NB_Pred,y_train)

#Test Data
B_NB_Pred_Test =B_NB.predict(x_test)
B_NB_Accu_Test = np.mean(B_NB_Pred_Test == y_test)
B_NB_Accu_Test #0.728  

pd.crosstab(B_NB_Pred_Test,y_test)