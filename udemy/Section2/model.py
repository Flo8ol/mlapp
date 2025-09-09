import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

chemin = "udemy/"

n_samples = 500
X1 = np.random.uniform(1,10,n_samples)
noise = np.random.normal(0,5,n_samples)
y = 10*X1+noise

data = pd.DataFrame({'Hours':X1, 'Test':y})

X=data[['Hours']]
y=data[['Test']]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()

model.fit(X_train_scaled, y_train)

print(model.coef_)
print(model.intercept_)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(mse, r2)

joblib.dump(model, chemin + 'linear_regression_model.pk1')
joblib.dump(scaler, chemin + 'scaler.pk1')