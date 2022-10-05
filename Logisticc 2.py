import matplotlib.pyplot as plt
import math
import numpy as np

def BMI_cal(h,w):
    h = h/100
    return w/(h**2)

def Loss_cal(p,y):
    Loss=0
    for i in range (len(p)):
        Loss += y[i] * math.log( p[i] ) + (1 - y[i]) * math.log(1 - p[i])
    return round(Loss,5)

def sigmoid(x):
    return round( 1/(1+np.exp(-x)),4)

def predict(w,h,b,weight,height):
    p=np.zeros(len(weight))
    for i in range(len(p)):
        p[i]=sigmoid(w*weight[i] +h*height[i] + b)
    return p

def predict_single_point(w,h,b):
    we,he=map(int,input("Nhap can nang va chieu cao: ").split())
    print('xac suat bao phi: ',sigmoid(w*we+h*he+b))

def accuracy(p,y):
    correct=0
    for i in range(len(y)):
        if round(p[i])==y[i]:
            correct+=1
    return round( correct/len(y),4 )

def min_max_scale(x):
    min_x,max_x=min(x),max(x)
    for i in range(len(x)):
        x[i]=(x[i]-min_x) /(max_x-min_x)
    return x

#18 entry                                                 10
weight=np.array([ 48, 65, 44, 55, 60, 52, 46, 73, 64, 55, 50, 57, 53, 69, 71, 81, 42, 61])
height=np.array([155,170,158,165,172,169,153,159,171,164,160,154,155,149,170,172,147,166])

y=np.array([0,0,0,0,1,0,1,1,0,0,0,1,0,1,1,1,0,0])


lr=0.000008
step,max_step=0,7000
w,h,b=0,0,0
while True:
    p=predict(w,h,b,weight,height)
    for i in range(len(weight)):
        w -= (p[i]-y[i])* weight[i]*lr
        h -= (p[i]-y[i])*height[i]*lr
        b -= (p[i]-y[i])*lr

    if step%500==0:
        print('step:',step,'Loss',Loss_cal(p,y))
    step += 1
    if step > max_step:
        break

print(round(w,5),round(h,5),round(b,5))
#.44 -.15 .03
print( 'accuracy score:', accuracy(p,y) )
BMI=[]
for i in range(len(weight)):
    BMI.append(BMI_cal(height[i],weight[i]))
    if round(p[i])!=y[i]:
     print('obese chance: ',round(BMI[i],4),p[i],'true value:',y[i])

predict_single_point(w,h,b)
