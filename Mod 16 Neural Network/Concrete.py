import numpy as np #scientific Calculation 
import pandas as pd #data manipulation 
from keras.models import Sequential #layer of neurons
from keras.layers import Dense # Connect the all the layers
import matplotlib.pyplot as plt #Data Visualization

#Loading the data
concrete = pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 16\\concrete.csv")

# Normalization function 
def norm_func(i):
     x = (i-i.min())/(i.max()-i.min())
     return (x)
concrete = norm_func(concrete) #applying the normalization on data

#spilting the data
predictors = concrete.iloc[:,0:8] #Creating the predictors variable from dataset
target = concrete.iloc[:,8] #Creating the target variable from dataset

##Partitioning the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size = 0.25) 

def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],kernel_initializer="normal",activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1]))
    model.compile(loss="mean_squared_error",optimizer="adam",metrics = ["accuracy"])
    return (model)

first_model = prep_model([8,50,100,1]) #layer of neuron
first_model.fit(np.array(x_train),np.array(y_train),epochs=500) #runnning epochs on Training data

pred_train = first_model.predict(np.array(x_train))  
pred_train = pd.Series([i[0] for i in pred_train])

rmse_value = np.sqrt(np.mean((pred_train-y_train)**2)) #calculating the RMSE value
rmse_value  

corrcoef_of_Training = np.corrcoef(pred_train,y_train) #checking the Correlation coefficients
corrcoef_of_Training 

#Visualising 
plt.plot(pred_train,y_train,"bo") #Plot to see the linear line on the training model

##Predicting on test data
pred_test = first_model.predict(np.array(x_test))
pred_test = pd.Series([i[0] for i in pred_test])

rmse_test = np.sqrt(np.mean((pred_test-y_test)**2)) #calculating the RMSE value
rmse_test #0.2894

corrcoef_of_Testing = np.corrcoef(pred_test,y_test) #checking the Correlation coefficients
corrcoef_of_Testing #0.9664

##Visualizing
plt.plot(pred_test,y_test,"bo") #Plot to see the linear line on the testing model 