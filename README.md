# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard python libraries for Gradient design.
2. Introduce the variables needed to execute the function.
3. Use function for the representation of the graph.
4. Using for loop apply the concept using the formulae.
5. Execute the program and plot the graph.
6. Predict and execute the values for the given conditions.

## Program:
```py
/*
Program to implement the linear regression using gradient descent.
Developed by: Dhayananth P S
RegisterNumber:212223040039  
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1.txt",header=None)

print("Profit prediction graph:")
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,1000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  """
  Take in a numpy array X,y,theta and generate the cost function in a linear regression model
  """
  m=len(y) #length of the training data
  h=X.dot(theta) #hypothesis
  square_err=(h-y)**2

  return 1/(2*m) * np.sum(square_err) #returning
  
  data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

print("Compute cost value:")
computeCost(X,y,theta) #Call the function

def gradientDescent(X,y,theta,alpha,num_iters):
  """
  Take in numpy array X,y and theta and update theta by taking number with learning rate of alpha

  return theta and the list of the cost of theta during each iteration
  """
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions -y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta,J_history  
  
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) value:")
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

print("Cost function using Gradient Descent graph:")
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

print("Profit prediction graph:")
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
  """
  Take in numpy array of x and theta and return the predicted value of y based on theta
  """

  predictions= np.dot(theta.transpose(),x)

  return predictions[0]
  
predict1=predict(np.array([1,3.5]),theta)*10000
print("Profit for the population 35,000:")
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("Profit for the population 70,000:")
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
![linear regression using gradient descent](sam.png)
## Profile prediction graph:
![Screenshot 2024-08-30 165208](https://github.com/user-attachments/assets/e5e16f9a-941d-4d5f-974a-faac7fbcec7b)
## Compute cost value:
![Screenshot 2024-08-30 165413](https://github.com/user-attachments/assets/535d3d92-3594-4ca8-8130-ef8e99b57be1)
## h(x) value:
![Screenshot 2024-08-30 165631](https://github.com/user-attachments/assets/c536c0f0-dee7-458a-98fb-9224e84f9d20)
## Cost function using Gradient Descent graph:
![Screenshot 2024-08-30 165807](https://github.com/user-attachments/assets/f04db727-86d4-4fbd-91df-90235efaf299)
## Profit prediction graph:
![Screenshot 2024-08-30 165931](https://github.com/user-attachments/assets/a06e90fb-7d9a-406d-8219-ec96015d04f3)
## Profit for the population 35,000:
![Screenshot 2024-08-30 170041](https://github.com/user-attachments/assets/e7c479e2-7a8a-4fb7-92c7-1cfa5d7566cf)
## Profit for the population 70,000:
![Screenshot 2024-08-30 170048](https://github.com/user-attachments/assets/c850bfac-89dd-4ea9-a709-1c98502ae72a)






## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
