import numpy as np
import random
import time
from multiprocessing import cpu_count
from joblib import Parallel, delayed

def sigmoid(z):
    return 1/(1+np.exp(-z))

def runEpoch(this, alpha, batch_size):
    random.shuffle(this.pairs)
    d_w = [np.zeros(w.shape) for w in this.weights]
    d_b = [np.zeros(b.shape) for b in this.biases]
    for i,pair in enumerate(this.pairs):
        delta_b, delta_w = this.backprop(pair[0],pair[1])
        for i in xrange(len(this.weights)):
            d_w[i] += delta_w[i]
            d_b[i] += delta_b[i]
        if (i+1)%batch_size == 0:
            for i in xrange(len(this.weights)):
                this.weights[i] -= (alpha/batch_size)*d_w[i]
                this.biases[i] -= (alpha/batch_size)*d_b[i]

class FeedForwardNeuralNetwork:
    def __init__(self, t):
        self.topology = t
        self.weights = []
        self.biases = []
        for i in range(1,len(t)):
            self.weights.append(np.random.uniform(low=-2/(t[i]+t[i-1]+1), high=\
            2/(t[i]+t[i-1]+1), size=(t[i-1],t[i])))
            self.biases.append(np.random.uniform(low=-2/(t[i]+t[i-1]+1), high=\
            2/(t[i]+t[i-1]+1), size=(t[i],1)))

    def feedfwd(self,_input):
        a = np.array(_input)
        a = a.reshape(a.shape[0],1)
        for w,b in zip(self.weights,self.biases):
            z = np.dot(w.transpose(),a)+b
            a = sigmoid(z)
        return a

    def backprop(self, _input, target):
        a = np.array(_input)
        a = a.reshape(a.shape[0],1)
        target = np.array(target)
        target = target.reshape(target.shape[0],1)
        a_list = [a]
        z_list = []
        for w,b in zip(self.weights,self.biases):
            z = np.dot(w.transpose(),a)+b
            a = sigmoid(z)
            a_list.append(a)
        d_E = a_list[-1]-target
        d_Y = a_list[-1]*(1-a_list[-1])
        d_b = d_E*d_Y
        d_w = np.dot(a_list[-2],d_b.transpose())
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w[-1] = d_w
        delta_b[-1] = d_b
        for i in range(2,len(self.weights)+1):
            d_b = np.dot(self.weights[-i+1],d_b)*a_list[-i]*(1-a_list[-i])
            d_w = np.dot(a_list[-i-1],d_b.transpose())
            delta_w[-i] = d_w
            delta_b[-i] = d_b
        return (delta_b,delta_w)

    def SGD(self, inputs, outputs, alpha, batch_size, epochs):
        self.pairs = zip(inputs,outputs)
        for i in xrange(epochs):
            random.shuffle(self.pairs)
            d_w = [np.zeros(w.shape) for w in self.weights]
            d_b = [np.zeros(b.shape) for b in self.biases]
            for i,pair in enumerate(self.pairs):
                delta_b, delta_w = self.backprop(pair[0],pair[1])
                for i in xrange(len(self.weights)):
                    d_w[i] += delta_w[i]
                    d_b[i] += delta_b[i]
                if (i+1)%batch_size == 0:
                    for i in xrange(len(self.weights)):
                        self.weights[i] -= (alpha/batch_size)*d_w[i]
                        self.biases[i] -= (alpha/batch_size)*d_b[i]

    def distSGD(self, inputs, outputs, alpha, batch_size, epochs):
        cores = cpu_count()
        self.pairs = zip(inputs,outputs)
        Parallel(n_jobs=cores)(delayed(runEpoch)(self, alpha, batch_size) for i in range(epochs))








if __name__ == "__main__":
    epochs = 1000000
    print "Performance Testing of dist and normal SGD on MLP with And, Or, Xor"
    test = FeedForwardNeuralNetwork([2,4,3])
    xs = [[0,0],[0,1],[1,0],[1,1]]
    ys = [[0,0,0],[0,1,1],[0,1,1],[1,1,0]]
    t0 = time.time()
    test.distSGD(xs, ys, 1, 1, epochs)
    print "Time to train {0} epochs using distributed SGD:".format(epochs)
    print time.time()-t0
    t0 = time.time()
    test.SGD(xs, ys, 1, 1, epochs)
    print "Time to train {0} epochs using distributed SGD:".format(epochs)
    print time.time()-t0
