import numpy as np
import matplotlib.pyplot as plt
from pylab import scatter, show, legend, xlabel, ylabel

def dataloader(file):
    with open(file,'r') as file:
        lines = file.readlines()
        
        dataList = []
        for line in lines:
            el = line.split('\n')[0].split(',')
            dataList.append(el)

    return  np.asarray(dataList,dtype=np.float32)

def dataextractor(datamatrix):
    X_rows,X_cols = datamatrix.shape
    X = datamatrix[:,0:-1]
    Y = datamatrix[:,-1]

    X = X.reshape((X_rows,X_cols-1))
    Y = Y.reshape((X_rows,1))

    ones_mat = np.ones((X_rows,1))
    X = np.append(ones_mat,X,axis=1)

    return X,Y

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def costFunctionReg(theta,X,y,lamb):
    
    h = sigmoid(np.dot(X,theta))
    theta[0] = 0
    J = (np.dot(y.T,log(h)) + np.dot((1 - y).T,log(1 - h)))/(-m) + (0.5*lamb/m) * sum(np.square(theta));

    temp = (np.dot((h - y).T,X).T/m) + np.dot(lamb/m,theta)
    grad = temp

    return J,grad

def plotter(X,Y):
    pos = np.where(Y == 1)
    neg = np.where(Y == 0)

    scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
    scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
    xlabel('Exam 1 score')
    ylabel('Exam 2 score')
    legend(['Not Admitted', 'Admitted'])
    show()

data = dataloader('ex2data1.txt')

X,Y = dataextractor(data)

plotter(X[:,1:3],Y)
