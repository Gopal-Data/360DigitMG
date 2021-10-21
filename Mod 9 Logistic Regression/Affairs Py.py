import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split  
from sklearn import metrics
from sklearn.metrics import classification_report

#loading the dataset
affair= pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 9\\affairs.csv")

#removing the unwanted feature
affair= affair.drop('Unnamed: 0', axis=1)
#checking for NaN value
affair.isna().sum()
# No NaN value found in the dataset
list(affair.columns) 

#Box Plot  
plt.boxplot(affair["naffairs"])
#outlier found and its not treated 
#all others variables are 1 and 0

#Coverting naffrairs varibles to 0 or 1
affair['naffairs'] = (affair['naffairs'] < 1).astype(int)

#Model 
model = sm.logit('naffairs ~ kids+ vryunhap+ unhap+ avgmarr+ hapavg+ vryhap+ antirel+ notrel+ slghtrel+ smerel+ vryrel+ yrsmarr1+ yrsmarr2+ yrsmarr3+ yrsmarr4+ yrsmarr5+ yrsmarr6', data = affair).fit()
model.summary2() # for AIC
#AIC 632.2126

model1 = sm.logit('naffairs ~ kids+ vryunhap+ unhap+ avgmarr+ hapavg+  antirel+ notrel+ slghtrel+ smerel+ yrsmarr1+ yrsmarr2+ yrsmarr3+ yrsmarr4+ yrsmarr5', data = affair).fit()
model1.summary2()
#AIC 632.2126 

model2 = sm.logit('naffairs ~ kids+ vryunhap+ unhap+ avgmarr+ hapavg+  antirel+ notrel+ slghtrel+ smerel+ yrsmarr5', data = affair).fit()
model2.summary2()
#AIC 630.3065

model3 = sm.logit('naffairs ~ kids+ vryunhap+ unhap+ antirel+ notrel+ slghtrel+ smerel+ yrsmarr5', data = affair).fit()
model3.summary2()
#AIC 634.7858 

model4= sm.logit('naffairs ~ kids+ vryunhap+ unhap + smerel+ yrsmarr5', data = affair).fit()
model4.summary2()
#AIC 641.5696 

model5 = sm.logit('naffairs ~ vryunhap+ unhap+  + smerel+ yrsmarr1', data = affair).fit()
model5.summary2()
#AIC 637.8768 

model6 = sm.logit('naffairs ~ kids+ vryunhap + slghtrel+ yrsmarr4', data = affair).fit()
model6.summary2()
#AIC 663.3865 

model7 = sm.logit('naffairs ~ vryunhap + slghtrel+ yrsmarr1+ yrsmarr2+ yrsmarr3+ yrsmarr4', data = affair).fit()
model7.summary2()
#AIC 660.8073

#data Spilting 
train_data, test_data = train_test_split(affair, test_size = 0.3)

# Model building
finalmodel = sm.logit('naffairs ~ kids+ vryunhap+ unhap+ avgmarr+ hapavg+  antirel+ notrel+ slghtrel+ smerel+ yrsmarr5', data = train_data).fit()
finalmodel.summary2()
#AIC 420.3078
##prediction values
train_pred = finalmodel.predict(train_data.iloc[ :, 1: ])


# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(420)

# taking threshold value as 0.5 and above the prob value will be treated as correct value 
train_data.loc[train_pred > 0.5, "train_pred"] = 1

# classification report
classification = classification_report(train_data["train_pred"], train_data["naffairs"])
classification

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['naffairs'])
confusion_matrx

accuracy_train = (25 + 301)/(420)
print(accuracy_train)
#0.7761

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(train_data["naffairs"], train_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc = metrics.auc(fpr, tpr)  
roc_auc #ROC 0.734

# Prediction on Test data set
test_pred = finalmodel.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(181)

# taking threshold value as 0.5 and above the prob value will be treated as correct value 
test_data.loc[test_pred > 0.5, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['naffairs'])
confusion_matrix

accuracy_test = (8 + 123)/(181) 
accuracy_test
#acc 0.7237

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["naffairs"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc = metrics.auc(fpr, tpr)  
roc_auc #0.6418