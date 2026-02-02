# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and prepare the multivariate input data and target values.
   
2.Initialize the SGD Regressor with appropriate learning rate and iterations.

3.Train the model using the given dataset and predict the output values.

4.Compare actual and predicted values using graphical visualization.
   

## Program:
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: SARAVANAN K
RegisterNumber:  25013282
*/
```
from sklearn.linear_model import SGDRegressor
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[1,2],[2,1],[3,4],[4,3],[5,5]])
y = np.array([5,6,9,10,13])
model = SGDRegressor(max_iter=1000, eta0=0.01, learning_rate='constant')
model.fit(X, y)

SGDRegressor
SGDRegressor(learning_rate='constant')
print("Weights:", model.coef_)
print("Bias:", model.intercept_)
Weights: [1.61566768 0.64128019]
Bias: [1.76711155]
y_pred = model.predict(X)
plt.scatter(y, y_pred)
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Actual vs Predicted (SGDRegressor)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Perfect prediction line
plt.show()
```


## Output:
<img width="349" height="61" alt="Screenshot 2026-02-02 150920" src="https://github.com/user-attachments/assets/8963572e-2cd6-4365-a6f4-b95ab424cc41" />
<img width="802" height="578" alt="Screenshot 2026-02-02 150903" src="https://github.com/user-attachments/assets/4fcef913-c5c1-4623-ae89-21af5d31df52" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
