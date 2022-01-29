import numpy as np
import pandas as pd
import sns as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression


def logistic(x, x0, k, L):
    return L/(1+np.exp(-k*(x-x0)))

model = LogisticRegression()

# Obese/not Obese: [list of weights in KGs]
data = {
    "Patient":[65, 75, 78, 85, 90],
    "Not Patient":[40, 45, 55, 70]
}

def plot_data(data):
    plt.figure(figsize=(8,6))
    plt.scatter(data["Patient"], [1]*len(data["Patient"]), s=200, c="red")
    plt.scatter(data["Not Patient"], [0]*len(data["Not Patient"]), s=200, c="green")
    plt.yticks([0, 1], ["Not Patient", "Patient"], fontsize=20)
    plt.ylim(-0.3, 1.2)
    plt.xlabel("Weight")

plot_data(data)
x = np.arange(39, 91, 0.5)
l = logistic(x, x0=65, k=0.5, L=1)
plt.plot(x,l, 'k:')
plt.grid()
plt.show()
