import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint


y0 = [10,1] # [fish, bears] units in hundreds

t = np.linspace(0,50,num=1000)

alpha = 1.1
beta = 0.4
delta = 0.1
gamma = 0.4

# steady state initial conditions
# y0 = [gamma/delta , alpha/beta] # [fish, bears] units in hundreds


params = [alpha, beta, delta, gamma]

def save_np_array(tmp1):
    with open("output.txt", "w") as f:
        for row in tmp1:
            np.savetxt(f, row)
    f.close()

def get_np_array():
    with open('output.txt','rb') as f:
        tmp1 = np.loadtxt(f).reshape(1000,2)

    return tmp1

def sim(variables, t, params):

    # fish population level
    x = variables[0]

    # bear population level
    y = variables[1]


    alpha = params[0]
    beta = params[1]
    delta = params[2]
    gamma = params[3]

    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y

    return([dxdt, dydt])


y = odeint(sim, y0, t, args=(params,))

#save_np_array(y) #<- salva array in txt
#y1 = get_np_array() #<- prendi array dal file txt 

#if(np.array_equal(y,y1)):
#    print("Corretti")

f,(ax1,ax2) = plt.subplots(2)

line1, = ax1.plot(t,y[:,0], color="b")



line2, = ax2.plot(t,y[:,1], color="r")

ax1.set_ylabel("Fish (hundreds)")
ax2.set_ylabel("Bears (hundreds)")
ax2.set_xlabel("Time")

plt.show()