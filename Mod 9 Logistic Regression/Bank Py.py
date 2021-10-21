import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split  
from sklearn import metrics
from sklearn.metrics import classification_report


#loading the dataset
bank= pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 9\\bank_data.csv")
 
#checking for NaN value
bank.isna().sum()
# No NaN value found in the dataset
list(bank.columns) 
bank.dtypes
 
#column rename
bank.rename(columns={'con_cellular': 'concellular', 'con_telephone': 'contelephone', 'con_unknown': 'conunknown', 'joadmin.': 'joadmin', 'joblue.collar': 'jobluecollar', 'joself.employed': 'joselfemployed'}, inplace=True)
 
#plots 
plt.boxplot(bank["y"])
 
#Model 
model = sm.logit('y ~ age +default +balance +housing +loan +duration + campaign +pdays +previous +poutfailure +poutother +poutsuccess +poutunknown +concellular +contelephone +conunknown +divorced +married +single +joadmin +jobluecollar +joentrepreneur +johousemaid +jomanagement +joretired +joselfemployed +joservices +jostudent +jotechnician +jounemployed +jounknown', data = bank).fit()
model.summary2()  
#AIC 22695.5408

model1 = sm.logit('y ~ age + balance + housing + loan + duration + campaign + divorced + married', data = bank).fit()
model1.summary2() 
#AIC 25683.7008

model2 = sm.logit('y ~ age + balance + housing + loan + duration + campaign + divorced + married', data = bank).fit()
model2.summary2() 
#AIC 25683.7008

model3 = sm.logit('y ~ balance+ housing+ loan+ duration+ campaign+ poutfailure+ poutsuccess +married +jostudent', data = bank).fit()
model3.summary2() 
#AIC 23460.7224   

model4 = sm.logit('y ~ balance+ housing+ loan+ duration+ campaign+ jostudent', data = bank).fit()
model4.summary2()     
#AIC 25692.4876   

model5 = sm.logit('y ~ age+ default+ balance+ housing+ loan+ duration+ campaign+ pdays+ previous+ poutfailure+ poutother+ poutsuccess+ poutunknown+ concellular+ contelephone+ conunknown+ divorced+ married+ single+ joadmin+ jobluecollar+ joentrepreneur+ johousemaid+ jomanagement+ joretired+ joselfemployed+ joservices+ jostudent+ jotechnician+ jounemployed+ jounknown', data = bank).fit()
model5.summary2()  
#AIC 22695.5408

model6 = sm.logit('y ~ age + balance + duration + campaign + previous + default + housing + loan + poutfailure + poutother + poutsuccess + concellular + contelephone + divorced + married + joadmin+ jobluecollar + johousemaid + jomanagement + joretired + jostudent + jotechnician + jounemployed', data = bank).fit()
model6.summary2()  
#AIC 22688.5616

model7 = sm.logit('y ~ age + balance + duration + campaign + previous + default + housing + loan + poutfailure + poutother + joadmin + jobluecollar + jostudent + jotechnician + jounemployed', data = bank).fit()
model7.summary2()  
#AIC  25116.7469

model8 = sm.logit('y~ balance + duration + campaign + previous + housing + loan + poutfailure + poutother + poutsuccess + concellular + contelephone + divorced + married + joadmin + jobluecollar + johousemaid + jomanagement + joretired + jostudent + jounemployed', data = bank).fit()
model8.summary2()  
#AIC 22686.3109

#data Spilting 
train_data, test_data = train_test_split(bank, test_size = 0.3)

# Model building
finalmodel = sm.logit('y~ balance + duration + campaign + previous + housing + loan + poutfailure + poutother + poutsuccess + concellular + contelephone + divorced + married + joadmin + jobluecollar + johousemaid + jomanagement + joretired + jostudent + jounemployed', data = train_data).fit()
finalmodel.summary2()
#AIC  16099.7063

##prediction values
train_pred = finalmodel.predict(train_data.iloc[ :, 1: ])


# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(31647)

# taking threshold value as 0.5 and above the prob value will be treated as correct value 
train_data.loc[train_pred > 0.5, "train_pred"] = 1

# classification report
classification = classification_report(train_data["train_pred"], train_data["y"])
classification

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['y'])
confusion_matrx

accuracy_train = (27301 + 1177)/(31647)
accuracy_train
#0.89986

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(train_data["y"], train_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc = metrics.auc(fpr, tpr)  
roc_auc #ROC 0.8868

# Prediction on Test data set
test_pred = finalmodel.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(13564)

# taking threshold value as 0.5 and above the prob value will be treated as correct value 
test_data.loc[test_pred > 0.5, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['y'])
confusion_matrix

accuracy_test = (11724+ 527)/(13564) 
accuracy_test
#acc 0.90319

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["y"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc = metrics.auc(fpr, tpr)  
roc_auc # 0.8868