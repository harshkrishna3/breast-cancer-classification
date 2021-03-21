from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd
from matplotlib import pyplot as plt

#importing data
data = pd.read_csv('data.csv')
print(data.head())

#cleaning the data
data = data.iloc[:, :-1]
data.dropna()


X = data.drop('diagnosis', axis=1)
y = pd.get_dummies(data['diagnosis'], drop_first=True)
print(X.iloc[:, 1])
print(y.head())

#visualize data
plt.scatter(X.iloc[:, 7], y)
plt.hist(X.iloc[:, 2])
plt.show()

#preprocessing
scalar = StandardScaler()
X = pd.DataFrame(scalar.fit_transform(X), columns = X.columns.values)
print(X.head())

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#train model
LogReg = LogisticRegression(solver='lbfgs')
model = LogReg.fit(X_train, y_train)

#checking accuracy
print(LogReg.score(X_test, y_test))
y_pred = LogReg.predict(X_test)
print(r2_score(y_test, y_pred))
