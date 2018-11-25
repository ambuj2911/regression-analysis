import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Social_Network_Ads.csv")

#dataset1=pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:,[2,3]].values
y= dataset.iloc[:,4].values

# splitting into train set and test set

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)

# scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression (random_state=0)
classifier.fit(x_train,y_train)

y_predict=classifier.predict(x_test)
print(y_predict)

from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test,y_predict)
cf

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
              alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
