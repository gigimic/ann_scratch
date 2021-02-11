# This is the simplest neural network from scratch
# No hidden layer is added
# Steps to follow:
# 1. Define independent variables and dependent variable
# 2. Define Hyperparameters
# 3. Define Activation Function and its derivative
# 4. Train the model
# 5. Make predictions

import numpy as np
#Independent variables
input_set = np.array([[0,1,0],
                      [0,0,1],
                      [1,0,0],
                      [1,1,0],
                      [1,1,1],
                      [0,1,1],
                      [0,1,0]])
#Dependent variable
labels = np.array([[1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1]])
labels = labels.reshape(7,1) #to convert labels to vector

np.random.seed(42)
weights = np.random.rand(3,1)
bias = np.random.rand(1)
lr = 0.05 #learning rate

# activation function - sigmoid 
def sigmoid(x):
    return 1/(1+np.exp(-x))

# function to calculate the derivative of the sigmoid function 
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

# train our ANN model with number of epoch = 2500
srno = [] 
errno = [] 
for epoch in range(5000):
    inputs = input_set
    XW = np.dot(inputs, weights)+ bias  #we find the dot product of the input and the weight and add bias to it
    z = sigmoid(XW)  #We pass the dot product through the sigmoid activation function
    error = z - labels # z is the predicted output
    # print(error.sum())
    dcost = error # cost function
    dpred = sigmoid_derivative(z)
    z_del = dcost * dpred # derivative of the cost fn is input x dcost x dpred
    inputs = input_set.T
    weights = weights - lr*np.dot(inputs, z_del) #weights are updated
    srno.append(epoch)
    errno.append(error.sum())
    # print(srno[epoch], errno[epoch])
    
    for num in z_del:
        bias = bias - lr*num # bias are updated

print(len(srno), len(errno))

# errors are plotted to find if the model is converging or the error is decreasing

# import matplotlib.pyplot as plt

# plt.scatter(srno, errno)
# plt.show()

# make prediction for case 1
single_pt = np.array([1,0,0])
result = sigmoid(np.dot(single_pt, weights) + bias)
print(result)

# make prediction for case 2
single_pt = np.array([0,1,0])
result = sigmoid(np.dot(single_pt, weights) + bias)
print(result)