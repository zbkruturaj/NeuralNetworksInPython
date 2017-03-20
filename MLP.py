import numpy as np
import pandas as pd
import random


# D_act = {'SIG':sigmoid}
# der = {'SIG':lambda x: sigmoid(x)*(1-sigmoid(x))}
		
class FeedForwardNetwork:
	def __init__(self, topology):

		self.layers = len(topology)
		self.topology = topology
		# self.cost = 
		# self.act_der = der[act]
		# self.act = self.sigmoid
		self.weights = [np.random.randn(y, x+1)/np.sqrt(x)
                        for x, y in zip(self.topology[:-1], self.topology[1:])]
		
	def sigmoid(self, x):
		return 1.0/(1.0+np.exp(x))

	def feedforward(self, inp):
		s = np.array([inp]).transpose()
		for w in self.weights:
			t = [np.ones(np.shape(s)[1])]
			s = np.concatenate((s, t))
			s = self.sigmoid(np.dot(w, s))
		return s
	
	def SGD(self, x_train, y_train, epochs, batch_size, learning_rate):
		# write SGD
		accY = []
		for e in range(epochs):
			indices = range(len(x_train))
			random.shuffle(indices)
			for j in range(len(x_train)/batch_size):
				dw =  [np.zeros((y, x+1))
                        		for x, y in zip(self.topology[:-1], self.topology[1:])]
				for k in range(batch_size):
					delta_w =  [np.zeros((y, x+1))
                        			for x, y in zip(self.topology[:-1], self.topology[1:])]
					tar = y_train[indices[j*batch_size+k]]
					inp = x_train[indices[j*batch_size+k]]
					act = []
					mult = []
					y = np.array([inp]).transpose()
					act.append(y)
					for w in self.weights:
						t = [np.ones(np.shape(y)[1])]
						y = np.concatenate((y, t))
						y = self.sigmoid(np.dot(w, y))
						act.append(y)
					t = [np.ones(np.shape(act[-2])[1])]
					t = np.concatenate((act[-2], t))
					delta_w[-1] = (tar-y)*np.dot(y,(1-y))*t.transpose()
					#print np.shape(delta_w[-1])
					#print np.shape(dw[-1])
					#print np.shape(self.weights[-1])
					for i in range(2,len(self.weights)+1):
						t = [np.ones(np.shape(act[-i])[1])]
						t1 = 1-act[-i]
						t1 = np.concatenate((t1, t)) # 1-H
						t = np.dot(delta_w[-i+1],t1) 
						#print np.shape(t)
						t = (self.weights[-i+1].transpose()*t)[:-1]
						#print np.shape(t)
						t2 = [np.ones(np.shape(act[-i-1])[1])]
						t2 = np.concatenate((act[-i-1], t2))
						#print np.shape(t2)
						delta_w[-i] = np.dot(t,t2.transpose())   #np.dot(delta_w[-i+1].transpose(),(1-act[-i]))*act[-i-1]
						#print np.shape(delta_w[-i])
						#print np.shape(dw[-i])
						#print np.shape(self.weights[-i])
					#return
					for i in range(0,len(self.weights)):
						dw[i] = np.add(dw[i],delta_w[i])
				for i in range(0,len(self.weights)):
					self.weights[i] = np.subtract(self.weights[i],learning_rate*dw[i])
			accY.append(self.accuracy(x_train, y_train))
	def accuracy(self, x_train, y_train):
		# Calculate the accuracy.
		acc = 0
		for i in range(len(x_train)):
			res = 1 if self.feedforward(x_train[i])[0][0] > 0.5 else 0
			if res == y_train[i]:
				acc += 1
		return acc*100.0/len(x_train)	

	def save(self, filename):
		# saveNN to a file for further recovery.
		pass
	def load(self, filename):
		# loads a saved NN into this object
		pass

seed = 102
np.random.seed(seed)
random.seed(seed)

a = 0.5
b = 0.6
r = 0.4

X = []
Y = []
for i in range(10000):
	x1 = random.random()
	x2 = random.random()
	X.append([x1,x2])
	Y.append(1 if (x1-a)*(x1-a)+(x2-b)+(x2-b) < r*r else 0)

f = FeedForwardNetwork([2,10,1])
print f.accuracy(X,Y)
f.SGD(X,Y,50,20,0.1)
print f.accuracy(X,Y)
print f.weights