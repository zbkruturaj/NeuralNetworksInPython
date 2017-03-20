import numpy as np
import random
import csv
np.random.seed(0)

## Initializing Weight

wx1 = random.random()
wx2 = random.random()
wh1 = random.random()
wh2 = random.random()
wb1 = random.random()
wb2 = random.random()

alpha = 0.1

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoidDerivative(x):
	return sigmoid(x)*(1-sigmoid(x))

def feedforward(x):
	x1,x2 = x[0],x[1]

	x10 = x1
	x20 = x2
	h10 = 0
	h20 = 0
	
	x11 = 0
	x21 = 0
	h11 = sigmoid(x10*wx1+h20*wh2+wb1)
	h21 = sigmoid(x20*wx2+h10*wh1+wb2)
	
	x12 = 0
	x22 = 0
	h12 = sigmoid(x11*wx1+h21*wh2+wb1)
	h22 = sigmoid(x21*wx2+h11*wh1+wb2)

	t1 = sigmoid(x12*wx1+h22*wh2+wb1)
	t2 = sigmoid(x22*wx2+h12*wh1+wb2)

	return (t1,t2)

def backpropagate(x,y):
	global wx1
	global wx2 
	global wb1 
	global wb2
	global wh1
	global wh2
	y1,y2 = y[0],y[1]
	x1,x2 = x[0],x[1]

	x10 = x1
	x20 = x2
	h10 = 0
	h20 = 0
	
	x11 = 0
	x21 = 0
	h11 = sigmoid(x10*wx1+h20*wh2+wb1)
	h21 = sigmoid(x20*wx2+h10*wh1+wb2)
	
	x12 = 0
	x22 = 0
	h12 = sigmoid(x11*wx1+h21*wh2+wb1)
	h22 = sigmoid(x21*wx2+h11*wh1+wb2)

	t1 = sigmoid(x12*wx1+h22*wh2+wb1)
	t2 = sigmoid(x22*wx2+h12*wh1+wb2)

	dwx11 = h11*(1-h11)*x10
	dwx12 = h22*(1-h22)*(wh1*dwx11)
	dwx13 = t1*(1-t1)*(x12+wh2*dwx12)
	dwx1 = (t1-y1)*dwx13

	dwx21 = h21*(1-h21)*x20
	dwx22 = h12*(1-h12)*(wh2*dwx21)
	dwx23 = t2*(1-t2)*(x22+wh1*dwx22)
	dwx2 = (t2-y2)*dwx23

	dwh11 = h22*(1-h22)*h11
	dwh12 = t1*(1-t1)*wh2*dwh11
	dwh1  = (t1-y1)*dwh12

	dwh21 = h12*(1-h12)*h21
	dwh22 = t2*(1-t2)*wh1*dwh21
	dwh2  = (t2-y2)*dwh22
	
	dwb11 = h11*(1-h11)
	dwb12 = h22*(1-h22)*(wh1*dwb11)
	dwb13 = t1*(1-t1)*(1+dwb12)
	dwb1 = (t1-y1)*dwb13

	dwb21 = h21*(1-h21)
	dwb22 = h12*(1-h12)*(wh2*dwb21)
	dwb23 = t2*(1-t2)*(1+dwb22)
	dwb2 = (t2-y2)*dwb23

	wx1 -= alpha*dwx1
	wx2 -= alpha*dwx2
	wb1 -= alpha*dwb1
	wb2 -= alpha*dwb2
	wh1 -= alpha*dwh1
	wh2 -= alpha*dwh2


## Get data
X = []
Y = []
with open('q3data.csv','rb') as csvfile:
	spr = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in spr:
		r = map(float,row[0].split(','))
		X.append((r[0],r[1]))
		Y.append((r[2],r[3]))

acc = 0
for x,y in zip(X,Y):
	t = feedforward(x)
	acc += 0.5*(t[0]-y[0])*(t[0]-y[0])
	acc += 0.5*(t[1]-y[1])*(t[1]-y[1])

accp = acc*100.00/(2*len(X))
print accp


X = []
Y = []
with open('foo.csv','rb') as csvfile:
	spr = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in spr:
		r = map(float,row[0].split(','))
		X.append((r[0],r[1]))
		Y.append((r[2],r[3]))
acc = 0
for x,y in zip(X,Y):
	t = feedforward(x)
	acc += 0.5*(t[0]-y[0])*(t[0]-y[0])
	acc += 0.5*(t[1]-y[1])*(t[1]-y[1])

accp = acc*100.00/(2*len(X))
print accp

for _ in range(1000):
	for x,y in zip(X,Y):
		backpropagate(x,y)

acc = 0
for x,y in zip(X,Y):
	t = feedforward(x)
	acc += 0.5*(t[0]-y[0])*(t[0]-y[0])
	acc += 0.5*(t[1]-y[1])*(t[1]-y[1])

accp = acc*100.00/(2*len(X))
print accp


X = []
Y = []
with open('q3data.csv','rb') as csvfile:
	spr = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in spr:
		r = map(float,row[0].split(','))
		X.append((r[0],r[1]))
		Y.append((r[2],r[3]))

acc = 0
for x,y in zip(X,Y):
	t = feedforward(x)
	acc += 0.5*(t[0]-y[0])*(t[0]-y[0])
	acc += 0.5*(t[1]-y[1])*(t[1]-y[1])

accp = acc*100.00/(2*len(X))
print accp























