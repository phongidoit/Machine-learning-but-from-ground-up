import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def loss_cal(w,b,x,y):
  Loss=0
  for i in range(10):
    Loss+= (y[i]-w*x[i]-b)**2
  return Loss


def round(x):
  return f'{x:.4f}'

w,b=0,0
x = [2, 3, 4, 5, 6, 7, 9, 10, 12, 12.5]
y = [3, 2, 5, 4, 7, 5.5, 8, 7.5, 9.5, 9]
Loss=loss_cal(w,b,x,y)

#pick anything you want as long as not too big or zero
learn_rate_w=0.00025
learn_rate_b=0.0008

#save the data for
log_step_w,log_step_b,log_Loss,log_step=[],[],[],[]

step=0
while True:
    #mathematic derivative thing
    dwloss,dbloss=0,0
    for i in range(10):
        dwloss += -2*y[i]*x[i] + 2*w*(x[i]**2) + 2*b*x[i]
        dbloss += -2*y[i] + 2*w*x[i] + 2*b
    dwloss/=10
    dbloss/=10
    Loss_new= loss_cal(w-learn_rate_w*dwloss,b-learn_rate_b*dbloss,x,y)

    #smaller the limit, the better but could lead to longer run
    limit_w,limit_b=0.000001,0.000001

    #if overshoot happen, auto reset with a smaller learning rate
    if Loss_new > Loss:
      learn_rate_w= 0.5*learn_rate_w
      learn_rate_b = 0.5 * learn_rate_b
      continue

    Loss_w=Loss_new
    Loss_b=Loss_new
    w=w-learn_rate_w*dwloss
    b=b-learn_rate_b*dbloss
    step+=1

    #take log of step to do visualization
    if ((step<100) or step%150==0) and step<20000:
      log_step.append(step)
      log_step_w.append(w)
      log_step_b.append(b)
      log_Loss.append(Loss_new)

    # when the change for w and b reach near the limit, let stop and call it a day
    if abs(learn_rate_w*dwloss) < limit_w and abs(learn_rate_b*dbloss)<limit_b:
      print("Number of step required:",step)
      break


#coeficient of the expression
print(f'{w:.4f}',f'{b:.4f}')

#
#everything below here is for animation only
#
fig, ax = plt.subplots()
fig.set_size_inches(8,8)

plt.plot(x, y,'o')

# Initial plot
x1 = np.arange(0., 10., 0.2)
y1 = np.arange(0., 10., 0.2)
line, = ax.plot(x1, y1)
plt.ylabel("y")
plt.xlabel("x")
title = ax.text(0.8,9.5,"")

def animate(i):
    title.set_text( ("step:",log_step[i],'w:',round(log_step_w[i]),
                    'b',round(log_step_b[i]),"Loss:",round(log_Loss[i])) )
    x1 = np.arange(0, 15, 0.05)
    line.set_xdata(x1)
    line.set_ydata(log_step_w[i] * x1 + log_step_b[i])
    return line,title,


def init():
    line.set_ydata(y)
    return line,


ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, len(log_Loss)),
                              init_func=init, interval=100, blit=True,repeat=False)
plt.show()



