

import pandas as pd
import pandas as Dataframe
from sklearn import linear_model
import statsmodels.api as sm

datos = pd.read_csv("barrios.csv")
dataframe = pd.DataFrame(datos)
print(datos)
X = (dataframe[["metros","barrio"]])
Y = (dataframe['precio'])

print(X)
print(Y)

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

metros = 200
barrio = 1
tipo = 1
print ('Precio aproximado de: \n', regr.predict([[metros ,barrio]]))

X = sm.add_constant(X)
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)
