#Multinomial Regression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the data set 
mdata = pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 10\\mdata.csv")
#removing the unwanted variables
mdata = mdata.drop(['Unnamed: 0','id','female'],axis=1 )
#creating Dummy variables for honors variables
mdata= pd.get_dummies(mdata, columns=['honors'])
mdata = mdata.drop(['honors_enrolled'], axis=1)
#renaming the variables
mdata = mdata.rename(columns={'honors_not enrolled': 'honors'})

#converting Catorical to Numberical data 
def score_to_numeric(x):
    if x=='high':
        return 3
    if x=='middle':
        return 2
    if x=='low':
        return 1
mdata['ses'] = mdata['ses'].apply(score_to_numeric)

#converting Catorical to Numberical data  
def score_to_numeric(x):
    if x=='private':
        return 2
    if x=='public':
        return 1
mdata['schtyp'] = mdata['schtyp'].apply(score_to_numeric)

mdata = mdata[['prog', 'ses', 'schtyp','read','write','math','science','honors']]

#checking for missing value in dataframe
mdata.isna().sum() 

#boxplot 
plt.boxplot(mdata["write"])
plt.boxplot(mdata["math"])
plt.boxplot(mdata["read"])
plt.boxplot(mdata["science"])

# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x = "prog", y = "math", data = mdata)
sns.boxplot(x = "prog", y = "read", data = mdata)
sns.boxplot(x = "prog", y = "write", data = mdata)
sns.boxplot(x = "prog", y = "science", data = mdata) 
  
# Correlation values between each independent features
mdata.corr()

train, test = train_test_split(mdata, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver="newton-cg").fit(train.iloc[:, 1:],train.iloc[:, 0])
 
# Test predictions
test_predict = model.predict(test.iloc[:, 1:]) 
#Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)
#0.55

#Train predictions 
train_predict = model.predict(train.iloc[:, 1:])

# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict)
# 0.65 