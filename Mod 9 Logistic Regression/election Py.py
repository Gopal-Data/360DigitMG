import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split  
from sklearn import metrics
from sklearn.metrics import classification_report

#loading the dataset
election= pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 9\\election_data.csv")
#checking for NaN value
election.isna().sum()
#removing NAN Values in dataset
election= election.drop([0, 0])

plt.boxplot(election["Result"])

list(election.columns) 
election.rename(columns={'Amount Spent':'Amount', 'Popularity Rank': 'Popularity'}, inplace=True)

election.corr()
#Model 
model = sm.logit('Result  ~ Year + Amount ', data = election).fit()
model.summary2()  
#AIC 15.8091

model1 = sm.logit('Result ~ Popularity', data = election).fit()
model1.summary2()  
#AIC 7.8191 

model2= sm.logit('Result ~ Year', data = election).fit()
model2.summary2()  
#AIC 14.7150 

model3= sm.logit('Result ~ Amount', data = election).fit()
model3.summary2()  
#AIC 16.5008

model4= sm.logit('Result ~ Amount + Popularity', data = election).fit()
model4.summary2()  
#AIC 9.8177 

model5= sm.logit('Result ~ Popularity', data = election).fit()
model5.summary2()  
#AIC 7.8191 

model6= sm.logit('Result ~ Year + Amount', data = election).fit()
model6.summary2()  
#AIC  15.8091

model7= sm.logit('Result ~ Year', data = election).fit()
model7.summary2()  
#AIC 14.7150 

#data Spilting 
train_data, test_data = train_test_split(election, test_size = 0.3)

# Model building
finalmodel = sm.logit('Result ~ Popularity', data = train_data).fit()
finalmodel.summary2()
#AIC 7.8191  
##prediction values
train_pred = finalmodel.predict(train_data.iloc[ :, 1: ])


# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(7)

# taking threshold value as 0.5 and above the prob value will be treated as correct value 
train_data.loc[train_pred > 0.5, "train_pred"] = 1

# classification report
classification = classification_report(train_data["train_pred"], train_data["Result"])
classification

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['Result'])
confusion_matrx

accuracy_train = (1 + 5)/(7)
print(accuracy_train)
#0.8571

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(train_data["Result"], train_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc = metrics.auc(fpr, tpr)  
roc_auc #ROC 0.9

# Prediction on Test data set
test_pred = finalmodel.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(3)

# taking threshold value as 0.5 and above the prob value will be treated as correct value 
test_data.loc[test_pred > 0.5, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Result'])
confusion_matrix

accuracy_test = ( 2+ 1)/(3) 
accuracy_test
#acc 1.0

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["Result"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc = metrics.auc(fpr, tpr)  
roc_auc # 1.0
