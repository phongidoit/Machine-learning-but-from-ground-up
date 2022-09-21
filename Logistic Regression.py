import numpy as np
import matplotlib.pyplot as plt
import random
import math

def sigmoid(s):
    return 1/(np.exp(-s)+1)

def probability(x,w):
    p=[]
    for i in range(len(x)):
        p.append(sigmoid((w[0]*x[i]+w[1])))
    return p


def Loss_cal(p,y):
    Loss=0
    for i in range(len(x)):
        Loss+= y[i]*math.log(p[i]) + (1-y[i])*math.log(1-p[i])
    return Loss

def to_percent(a,b):
    return round(a/len(b),5)

def accuracy(y,p):
    #4 true/false positive/negative upgrade later
    true_pos,true_neg,false_pos,false_neg=0,0,0,0
    for i in range(len(y)):
        if y[i] == 1:
            if p[i] <0.5:
                false_pos+=1
            else:
                true_pos+=1
        else:
            if p[i] <= 0.5:
                true_neg+=1
            else:
                false_neg+=1
    return to_percent(true_pos+true_neg,y)


#create data
# array of 35 integer, ranging from 1-30
x=np.random.randint(30,size=35)
y=[]
p=[]
for num in x:
    val=0
    if num>13:
        if (random.random()>0.1):
            val=1
    else:
        if (random.random()>0.9):
            val=1
    y.append(val)

#training model
Loss=0
lr=0.0003
w=[0,0]
max_step=9000
step=0

while True:
    p=probability(x,w)
    Loss=Loss_cal(p,y)
    p=probability(x,w)
    #derrivative time
    for i in range (len(x)):
        w[0] += lr * (y[i]-p[i])*x[i]
        w[1] += lr* (y[i]-p[i])
    step+=1
    if (step%100==0):
        score=accuracy(y,p)
        print('step:',step,'Loss',Loss)
    if step> max_step:
        break

score=accuracy(y,p)
print('accuracy',score)

#drawing time
x1=np.linspace(-2,32,1000)
y1=sigmoid(w[0]*x1+w[1])
plt.plot(x,y,'o')
plt.plot(x1,y1,'g-',linewidth=2)
plt.show()

