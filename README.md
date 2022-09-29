# Experiment-2--Implementation-of-Perceptron
##AIM:

To implement a perceptron for classification using Python

EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one 
f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’

A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.


ALGORITHM:
Importing the libraries
Importing the dataset
Plot the data to verify the linear separable dataset and consider only two classes
Convert the data set to scale the data to uniform range by using Feature scaling

Split the dataset for training and testing
Define the input vector ‘X’ from the training dataset
Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2
Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
Assign the learning rate
For ‘N ‘ iterations ,do the following:
        v(i) = w(i)*x(i)
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)
Plot the error for each iteration 
Print the accuracy


 PROGRAM:
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron:
 def _init_(self, learning_rate=0.1):
   self.learning_rate = learning_rate
   self._b = 0.0  # y-intercept
   self._w = None  # weights assigned to input features
   self.misclassified_samples = []
 def fit(self, x: np.array, y: np.array, n_iter=10):
   self._b = 0.0
   self._w = np.zeros(x.shape[1])
   self.misclassified_samples = []
   for _ in range(n_iter):
     # counter of the errors during this training iteration
     errors = 0
     for xi, yi in zip(x, y):
       update = self.learning_rate * (yi - self.predict(xi))
       self._b += update
       self._w += update * xi
       errors += int(update != 0.0)
     self.misclassified_samples.append(errors)
 def f(self, x: np.array) -> float:
   return np.dot(x, self._w) + self._b
 def predict(self, x: np.array):
   return np.where(self.f(x) >= 0, 1, -1)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
df.head()
y = df.iloc[:, 4].values
x = df.iloc[:, 0:3].values
x = x[0:100, 0:2]  # reduce the dimensionality of the data
y = y[0:100]
plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x',
           label='Versicolour')

plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.legend(loc='upper left')

plt.show()

y = np.where(y == 'Iris-setosa', 1, -1)



x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()


from matplotlib.colors import ListedColormap

def plot_decision_regions(x, y):
   resolution = 0.001
   markers = ('o', 'x')
   cmap = ListedColormap(('red', 'blue'))
   x1_min, x1_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
   x2_min, x2_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
   Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
   Z = Z.reshape(xx1.shape)
   plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
   plt.xlim(xx1.min(), xx1.max())
   plt.ylim(xx2.min(), xx2.max()) 
   for idx, c1 in enumerate(np.unique(y)):
       plt.scatter(x=x[y == c1, 0],
                   y=x[y == c1, 1], 
                   alpha=0.8, 
                   c=cmap(idx), 
                   marker=markers[idx], 
                   label=c1)
   plt.show()

plot_decision_regions(x_test, y_test)

 
Output:



![op1](https://user-images.githubusercontent.com/83326978/193039675-9e754f1d-23d7-47a6-b745-a7cbc675571a.png)











![op2](https://user-images.githubusercontent.com/83326978/193039702-7866cfac-d0de-46de-af55-4a989d768b61.png)






Result:
Thus,a perceptron for classification using Python is executed successfully
