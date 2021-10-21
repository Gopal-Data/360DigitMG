import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import train_test_split

diabetes= pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 14\\Diabetes.csv")
diabetes.info()
 
# Dummy variables
diabetes.head()
list( diabetes)

#rename columns in Dataframe
diabetes.columns=['pregnant','glucose','pressure','thickness','insulin','index','pedigree','Age','variable']
list(diabetes)

diabetes = pd.get_dummies(diabetes, columns = ["variable"], drop_first = True)
#diabetes=diabetes.iloc[:,[8,0,1,2,3,4,5,6,7]]   
colname = list (diabetes.columns) 

predictors = colname[:8]
target = colname[8]

#Model building
train,test=train_test_split(diabetes,test_size=0.2)
model=DT(criterion='gini')
model.fit(train[predictors],train[target])

#prediction on the test data 
preds =model.predict(test[predictors])
pd.crosstab(test[target],preds,rownames=['Actual'], colnames=['predictors'])
#test data accuracy 
np.mean(preds==test[target])
#0.675
    
#prediction on the train data  
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames=['Actual'], colnames=['predictors'])

#test data accuracy 
np.mean(preds==train[target])
#1.0 

#Model building for purning
train,test=train_test_split(diabetes,test_size=0.2)
model=DT(criterion='gini', max_depth=4)
model.fit(train[predictors],train[target])

#prediction on the test data 
preds = model.predict(test[predictors])
pd.crosstab(test[target],preds,rownames=['Actual'], colnames=['predictors'])
#test data accuracy 
np.mean(preds==test[target])
#0.75 max = 5  #0.73 max = 6 #0.67 max = 7 #0.766 max = 4  #0.727 3
#prediction on the train data  
preds1 = model.predict(train[predictors])
pd.crosstab(train[target], preds1, rownames=['Actual'], colnames=['predictors'])

#test data accuracy 
np.mean(preds1==train[target])
#0.827 5  #0.840 6 #0.895 7 #0.79 4 #0.765 3 


# Pruning the Tree
# Minimum observations at the internal node approach
train,test=train_test_split(diabetes,test_size=0.2)
modelmin=DT(criterion='gini', min_samples_split = 5)
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
# 0.78

# Minimum observations at the internal node approach
train,test=train_test_split(diabetes,test_size=0.2)
modelleaf=DT(criterion='gini', min_samples_leaf = 5)
modelleaf.fit(train[predictors],train[target])

#prediction on the test data 
predsleaf = modelleaf.predict(test[predictors])
pd.crosstab(test[target],predsleaf,rownames=['Actual'], colnames=['predictors'])
#test data accuracy 
np.mean(predsleaf==test[target]) #0.72

#prediction on the train data  
predsleaf1 = model.predict(train[predictors])
pd.crosstab(train[target], predsleaf1, rownames=['Actual'], colnames=['predictors'])

#test data accuracy 
np.mean(predsleaf1==train[target]) #0.78
