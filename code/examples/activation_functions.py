import numpy as np
import matplotlib.pyplot as plt

#https://vidyasheela.com/post/relu-activation-function-with-python-code

# Rectified Linear Unit (ReLU)
def ReLU(x):
  data = [max(0,value) for value in x]
  return np.array(data, dtype=float)

# Derivative for ReLU
def der_ReLU(x):
  data = [1 if value>0 else 0 for value in x]
  return np.array(data, dtype=float)

# Sigmoid Activation Function
def sigmoid(x):
  return 1/(1+np.exp(-x))



# Generating data for Graph
x_data = np.linspace(-10,10,100)
y_data = ReLU(x_data)
#dy_data = der_ReLU(x_data)

# Graph
#plt.plot(x_data, y_data, x_data, dy_data)
plt.plot(x_data, y_data)
plt.title('ReLU Activation Function')
plt.legend(['ReLU'])
plt.grid()
plt.show()



# Generating data to plot
x_data = np.linspace(-10,10,100)
y_data = sigmoid(x_data)

data = {
    "Patient":[3, 4, 6, 8, 9],
    "Not Patient":[ -3, -4, -6, -8, -9]
}


# Plotting
plt.plot(x_data, y_data)
plt.scatter(data["Patient"], [1] * len(data["Patient"]), s=200, c="red")
plt.scatter(data["Not Patient"], [0] * len(data["Not Patient"]), s=200, c="green")

plt.title('Logistic Regression ')
plt.legend(['logistic'])
plt.grid()
plt.show()