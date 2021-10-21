import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
 
fraud = pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\Mod 14\\Fraud_check.csv")
fraud.info()
 
# Dummy variables
fraud.head()
list(fraud)
#rename columns in Dataframe
fraud = fraud.rename(columns={'Taxable.Income':'Taxable','City.Population':'Population' , 'Work.Experience':'Work', 'Marital.Status' : 'Marital'})

#typcasting sales variables
lb= LabelEncoder()
fraud["Undergrad"] = lb.fit_transform(fraud["Undergrad"]) 
fraud["Marital"] = lb.fit_transform(fraud["Marital"]) 
fraud["Urban"] = lb.fit_transform(fraud["Urban"])
str(fraud.Urban)

fraud.loc[fraud.Taxable <= 30000, "Taxable"]= "1"
fraud.loc[fraud.Taxable !='1', "Taxable"]= "0"
fraud=fraud.iloc[:,[0,1,3,4,5,2]]   

colname = list(fraud.columns) 

predictors = colname[:5]
target = colname[5]

#Model building
train,test=train_test_split(fraud,test_size=0.2)
model=DT(criterion='entropy')
model.fit(train[predictors],train[target])

#prediction on the test data 
preds =model.predict(test[predictors])
pd.crosstab(test[target],preds,rownames=['Actual'], colnames=['predictors'])
#test data accuracy 
np.mean(preds==test[target])
#0.66
    
#prediction on the train data  
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames=['Actual'], colnames=['predictors'])

#test data accuracy 
np.mean(preds==train[target])
#1.0

#Model building for purning
train,test=train_test_split(fraud,test_size=0.2)
model=DT(criterion='entropy', max_depth=3)
model.fit(train[predictors],train[target])

#prediction on the test data 
preds = model.predict(test[predictors])
pd.crosstab(test[target],preds,rownames=['Actual'], colnames=['predictors'])
#test data accuracy 
np.mean(preds==test[target])
#0.825 max = 3 
#prediction on the train data  
preds1 = model.predict(train[predictors])
pd.crosstab(train[target], preds1, rownames=['Actual'], colnames=['predictors'])

#test data accuracy 
np.mean(preds1==train[target])
#0.7875  max = 3  

# Pruning the Tree
# Minimum observations at the internal node approach
train,test=train_test_split(fraud,test_size=0.2)
modelmin=DT(criterion='entropy', min_samples_split = 15)
modelmin.fit(train[predictors],train[target])

#prediction on the test data 
predsmin = modelmin.predict(test[predictors])
pd.crosstab(test[target],predsmin,rownames=['Actual'], colnames=['predictors'])
#test data accuracy 
np.mean(predsmin==test[target])
#0.683

#prediction on the train data  
predsmin1 = model.predict(train[predictors])
pd.crosstab(train[target], predsmin1, rownames=['Actual'], colnames=['predictors'])

#test data accuracy 
np.mean(predsmin1==train[target])
# 0.80

# Minimum observations at the internal node approach
train,test=train_test_split(fraud,test_size=0.2)
modelleaf=DT(criterion='entropy', min_samples_leaf = 10)
modelleaf.fit(train[predictors],train[target])

#prediction on the test data 
predsleaf = modelleaf.predict(test[predictors])
pd.crosstab(test[target],predsleaf,rownames=['Actual'], colnames=['predictors'])
#test data accuracy 
np.mean(predsleaf==test[target])
#0.70

#prediction on the train data  
predsleaf1 = model.predict(train[predictors])
pd.crosstab(train[target], predsleaf1, rownames=['Actual'], colnames=['predictors'])

#test data accuracy 
np.mean(predsleaf1==train[target]) #0.789
