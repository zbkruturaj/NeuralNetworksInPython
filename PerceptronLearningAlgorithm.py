import numpy as np
import pandas as pd
import random

def eval_perceptron(inp, w):
    return sgn(mult(inp,w))

def mult(x,y):
    res = 0.0
    for (e,f) in zip(x,y):
        res += e*f
    return res

def smult(a,x):
    t = []
    for i in range(len(x)):
        t.append(a*x[i])
    return t

def add(x,y):
    res = []
    for (e,f) in zip(x,y):
        res.append(e+f)
    return res

def sgn(x):
	return 1 if x > 0 else -1 if x < 0 else 0 

def conv(df, head):
	df[head] = df[head].astype('category')
	df[head].cat.categories = range(len(df[head].cat.categories))
	df[head] = df[head].astype('float64')


headers = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]
df = pd.read_csv('bank.csv', names=headers)
total_records = len(df)
c_h = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "duration", "previous", "poutcome"]
map(lambda x:conv(df,x), c_h)
df["y"] = df["y"].astype('category')
df["y"].cat.categories = [-1,1]
df["y"] = df["y"].astype('float64')
msk = np.random.randn(total_records) <= 0.5

df_tr = df.sample(frac=0.5,random_state=627)
df_te = df.drop(df_tr.index)

tar_tr = df_tr['y']
tar_te = df_te['y']


del df_tr['y']
del df_te['y']



## l = df_tr.values.tolist()
## tl = tar_tr.tolist()

## print l[0]
## print tl[0]
d = (df_tr - df_tr.mean() / (df_tr.max()-df_tr.min()))
l_tr = d.values.tolist()
t_tr = tar_tr.tolist()

d = (df_te - df_te.mean() / (df_te.max()-df_te.min()))
l_te = d.values.tolist()
t_te = tar_te.tolist()
## print d.head()

inputs = l_tr
outputs = t_tr
learning_rate = 0.1
n_inputs = len(inputs[0])
epochs = 100
w = []
r = random.Random(21)#2612)
for i in range(n_inputs):
    w.append(r.random()*2-1)
print w

score = 0.0
for (inp,out) in zip(l_te,t_te):
	if out == eval_perceptron(inp,w):
		score += 1.0

print score*100/len(t_te)

for i in range(epochs):
    for (inp,out) in zip(inputs,outputs):
        pred = eval_perceptron(inp,w)
        w = add(w,smult(learning_rate*(out-pred),inp))
print w

score = 0.0
for (inp,out) in zip(l_te,t_te):
	if out == eval_perceptron(inp,w):
		score += 1.0

print score*100/len(t_te)

score = 0.0
for (inp,out) in zip(l_tr,t_tr):
	if out == eval_perceptron(inp,w):
		score += 1.0

print score*100/len(t_tr)

