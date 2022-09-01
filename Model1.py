import imp

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from numba import jit, cuda
import imageio

import time
curr = time.time()

pic = imageio.imread(r"C:\Users\Lakshay Sharma\Pythom\PracticeML\A.jpg")

plt.figure()
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
Data = pd.read_excel(r"C:\Users\Lakshay Sharma\Pythom\PracticeML\GATES.xlsx")
#print(Data[["A","B"]].values)                              #2d array           if you remove values returns dict
dataAB = Data[["A","B"]].values


def f(w,b,x):
    return 1.0 / (1.0 + np.exp(-(w*x + b)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def clean(list):
    refinedList = []
    for i in list:
        if type(i) == MetaNeuron:
            refinedList.append(i.y)
        else:
            refinedList.append(i)
    return refinedList

def grad_b(w,b,x,y):
    fx = f(w,b,x)
    return (fx - y) * fx * (1 - fx)

class MetaNeuron:
    def __init__(self):
        self.b = 0
        self.W = []
        self.X = []
        self.y = 0
    
    def update(self):
        self.W = list(np.random.rand(len(self.X)))

    def info(self, X=1, W=0, B=0, Y=1):
        temp = {}
        if X:   temp['X'] = clean(self.X)
        if W:   temp['W'] = self.W
        if B:   temp['B'] = self.b
        if Y:   temp['Y'] = self.y
        return temp

    def comp(self):
        self.y = sigmoid(np.matmul(np.array(self.W).T, clean(self.X))+self.b)


class MetaNetwork:
    def __init__(self, dummyData, hiddenLayerSize, outputLayerSize):
        self.inputLayerSize = len(dummyData)
        self.Data = dummyData
        self.hiddenLayerSize = tuple(list(hiddenLayerSize))
        self.size = [self.inputLayerSize]+list(self.hiddenLayerSize)+[outputLayerSize]
        self.size = tuple(self.size)
        print("Network Created:\t",self.size)
        # Neurons Created
        self.InputLayer = [MetaNeuron() for i in range(self.inputLayerSize)]
        self.OutputLayer = [MetaNeuron() for o in range(outputLayerSize)]
        self.HiddenLayer = [[MetaNeuron() for i in range(hiddenLayerSize[j])] for j in range(len(self.hiddenLayerSize))]
        self.Layers = [self.InputLayer] + self.HiddenLayer + [self.OutputLayer] 
        self.connect() 

    def connect(self):
        # input Layer to Data
        for n,i in enumerate(self.Layers[0]):
            i.X.append(self.Data[n])
            i.update()
        # rest all layers
        for i in range(1, len(self.size)):                  #   every layer (from second)
            for j in range(self.size[i]):                   #   every element
                for k in range(self.size[i-1]):             #   with previous layer   = i-1
                    self.Layers[i][j].X += [self.Layers[i-1][k]]
                self.Layers[i][j].update()
          
    def show(self, info = False):
        for u in range(self.size[0]):                       #   first layer
            if info:
                pointname = self.Layers[0][u].info()
                plt.text(0,u, pointname)
        for i in range(1, len(self.size)):                  #   every layer (from second)
            for j in range(self.size[i]):                   #   every element
                for k in range(self.size[i-1]):             #   with previous layer   = i-1
                    if info:
                        pointname = self.Layers[i][j].info(0,1,0,0)
                        plt.text(i,j, pointname)
                    plt.plot([i-1,i],[k,j], linestyle="-", marker="D",color='red',linewidth=self.Layers[i][j].W[k]*5)
        plt.show()

    def run(self):
        for i in self.Layers:               
            for j in i: 
                j.comp()
        """
        #input layer
        for i in range(self.inputLayerSize):
            x.append(1)
            y.append(i)
        #hidden layer
        for i in range(self.hiddenLayerDimensions[1]):
            for j in range(self.hiddenLayerDimensions[0]):
                x.append(x[-i+1]+i+1)
                y.append(j)
        #output layer
        for i in range(self.outputLayerSize):
            x.append(x[-1-i]+1)
            y.append(i)
        plt.plot(x,y,linestyle = "solid",marker = "o",color="red")
        plt.pause(5)
"""
    

Network1 = MetaNetwork(pic[0],(10,10),1)
Network1.run()

print("Runtime: \t", time.time() - curr)
#Network1.show(1)



