import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#dataset=pd.read_csv("Social_Network_Ads.csv")

dataset=pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:,[0]].values
y= dataset.iloc[:,1].values

# splitting into train set and test set

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(x_train, y_train)  

y_predict=regressor.predict(x_test)
print(y_predict)

#df = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})  
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))  
np.sqrt(metrics.mean_squared_error(y_test, y_predict))

plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,regressor.predict(x_train))

plt.xlabel("Exp")
plt.ylabel("salary")
plt.title("lineaar reg for sal_test and exp_test")
plt.show()
