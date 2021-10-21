import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#loading the dataset 
company = pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 14\\company.csv")
list(company)

#plot
plt.hist(company.Sales)
plt.hist(company.Income) 
plt.hist(company.Price) #Price has a normal distribuction 
plt.hist(company.Advertising) #Most of the computer has no Advertisment 

#typcasting sales variables
company.loc[company.Sales <= 10, "Sales"]= "1"
company.loc[company.Sales != '1', "Sales"]= "0"
 
# n-1 dummy variables will be created for n categories
lb= LabelEncoder()
company["ShelveLoc"] = lb.fit_transform(company["ShelveLoc"]) 
company["Urban"] = lb.fit_transform(company["Urban"])
company["US"] = lb.fit_transform(company["US"])


         
#spilting the data
colname = list(company.columns)
predictors = colname[1:11]
target = colname[0]

#Model building
train,test=train_test_split(company,test_size=0.2)
model=DT(criterion='entropy')
model.fit(train[predictors],train[target])

#prediction on the test data 
preds = model.predict(test[predictors])
pd.crosstab(test[target],preds,rownames=['Actual'], colnames=['predictors'])
#test data accuracy 
np.mean(preds==test[target])
#0.862
    
#prediction on the train data  
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames=['Actual'], colnames=['predictors'])

#test data accuracy 
np.mean(preds==train[target])
#1.0 

#Model building for Pruning 
train,test=train_test_split(company,test_size=0.2)
model=DT(criterion='entropy', max_depth=4)
model.fit(train[predictors],train[target])

#prediction on the test data 
preds = model.predict(test[predictors])
pd.crosstab(test[target],preds,rownames=['Actual'], colnames=['predictors'])
#test data accuracy 
np.mean(preds==test[target]) #0.8375 max = 5  .8 = 3  0.8375 =4
#prediction on the train data  
preds1 = model.predict(train[predictors])
pd.crosstab(train[target], preds1, rownames=['Actual'], colnames=['predictors'])

#train data accuracy 
np.mean(preds1==train[target])
#0.91875 max = 5 0.85 =3 0.88 =4

#above max depth =5 its moving the test and train accuracy away from each scorce 

# Pruning the Tree
# Minimum observations at the internal node approach
train,test=train_test_split(company,test_size=0.2)
modelmin=DT(criterion='entropy', min_samples_split = 15)
modelmin.fit(train[predictors],train[target])

#prediction on the test data 
predsmin = modelmin.predict(test[predictors])
pd.crosstab(test[target],predsmin,rownames=['Actual'], colnames=['predictors'])
#test data accuracy 
np.mean(predsmin==test[target])
#0.75

#prediction on the train data  
predsmin1 = model.predict(train[predictors])
pd.crosstab(train[target], predsmin1, rownames=['Actual'], colnames=['predictors'])

#test data accuracy 
np.mean(predsmin1==train[target])
# 0.91 
#tried with 3 ,5 10 and 15 and the test accuracy remained around .75 to .77 train accuracy was above .90

# Minimum observations at the internal node approach
train,test=train_test_split(company,test_size=0.2)
modelleaf=DT(criterion='entropy', min_samples_leaf = 10)
modelleaf.fit(train[predictors],train[target])

#prediction on the test data 
predsleaf = modelleaf.predict(test[predictors])
pd.crosstab(test[target],predsleaf,rownames=['Actual'], colnames=['predictors'])
#test data accuracy 
np.mean(predsleaf==test[target])
#0.825 leaf = 4  #0.8375 = 5  #0.775 = 10 

#prediction on the train data  
predsleaf1 = model.predict(train[predictors])
pd.crosstab(train[target], predsleaf1, rownames=['Actual'], colnames=['predictors'])

#test data accuracy 
np.mean(predsleaf1==train[target]) #0.9125 = 4  0.91875  = 5   0.91875 = 10  