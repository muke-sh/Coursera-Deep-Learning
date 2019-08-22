import numpy as np
import matplotlib.pyplot as plt

def data_loader(file):
    '''data_loader() module loads a file and return a numpy array.'''
    data = open(file,'r')
    
    #read file content in a list
    file_data = data.readlines()
    data_list = []

  #  print(file_data)
    for i in range(len(file_data)):
        NewTuple = []

        InterList = (((file_data[i]).split('\n'))[0]).split(',')
        for i in range(len(InterList)):
            el1 = float(InterList[i])
           # print(float(el1))
            NewTuple.append(el1)
        data_list.append(NewTuple)
    
    data_matrix = np.asarray(data_list)
    
    return data_matrix
    
def data_extractor(data_matrix):
    
    X = data_matrix[:,0:-1]
    Y = data_matrix[:,-1]
    
    X_rows,X_cols = data_matrix.shape

    X = X.reshape((X_rows,X_cols-1))
    Y = Y.reshape((X_rows,1))
    ones_mat = np.ones((X.shape[0],1))
    X = np.append(ones_mat,X,axis=1)
   
    return X,Y

def plotter(X,Y):
    plt.plot(X,Y,'rx')
    plt.xlabel('Population')
    plt.ylabel('Revenue')
    plt.xticks([])
    plt.yticks([])
    plt.title('Data Scatter')
    plt.show()

def Compute_cost(X,y,theta):
    m = X.shape[0]
    h = np.dot(X,theta)
    error = h-y
    squared_error = np.square(error)/(2*m)
    cost = np.sum(squared_error,axis=0)
    print(cost)
    return cost

def gradien_descent(X,Y,theta,alpha,num_iter):
    m = len(Y)
    iter = []
    J_hist = []
    
    for i in range(num_iter):
        h = np.dot(X,theta)
        error = h - Y

        derivative =  (alpha/m) * error 

        temp0 = theta[0] -np.dot(derivative.T,X[:,0])
        temp1 = theta[1] -np.dot(derivative.T,X[:,1])

        theta[0] = temp0
        theta[1] = temp1
        cost = Compute_cost(X,Y,theta)
        J_hist.append(cost)
        iter.append(i)

    return iter,J_hist,theta     



if __name__ == '__main__':
    data_loader('ex1data1.txt')
    X,y = data_extractor(data_loader('ex1data1.txt'))


    #plotter(X[:,-1],y)
    #theta = np.array([[0],[0]])
    theta = np.random.randn(2,1)
    plt.figure(1)
    plt.plot(X[:,-1],np.dot(X , theta))
    plt.figure(2)
    iter,J_hist,theta2 = gradien_descent(X,y,theta,0.01,2000)
    plt.plot(iter,J_hist,'-')
    plt.xlabel("No. of iterations")
    plt.ylabel("Cost")
    plt.figure(3)
    plt.plot(X[:,-1],np.dot(X , theta2))
    plt.show()