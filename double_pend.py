import scipy.integrate as integrate
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# mass, length
m1 = 1
m2 = 1
l1 = 1
l2 = 1

# grav
g = 9.81

# timestep
dt = 0.02
    # tot length
T = 100
t = np.arange(0.0, T, dt)

# initial angles
th1 = 90
th2 = 180
# initial angular velocities
w1 = 0
w2 = 0

# trail length
N = 12

def derivs(state, t):
    dydx = np.zeros_like(state)

    dydx[0] = state[1]
    delta = state[2] - state[0]

    den1 = (m1 + m2)*l1 - m2*np.cos(delta)*np.cos(delta)
    dydx[1] = ( m2*l1*state[1]*state[1]*np.sin(delta)*np.cos(delta)
                + m2*g*np.sin(state[2])*np.cos(delta)
                + m2*l2*state[3]*state[3]*np.sin(delta)
                - (m1 + m2)*g*np.sin(state[0])
                )/den1

    dydx[2] = state[3]

    den2 = (l2/l1)*den1
    dydx[3] = ( - m2*l2*state[3]*state[3]*np.sin(delta)*np.cos(delta)
                + (m1 + m2)*g*np.sin(state[0])*np.cos(delta)
                - (m1 + m2)*l1*state[1]*state[1]*np.sin(delta)
                - (m1 + m2)*g*np.sin(state[2])
                )/den2
    return dydx

state = np.radians([th1, w1, th2, w2])

y = integrate.odeint(derivs, state, t)

x1 = l1*np.sin(y[:,0])
y1 = -l1*np.cos(y[:,0])

x2 = l2*np.sin(y[:,2]) + x1
y2 = -l2*np.cos(y[:,2]) + y1

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on = False, xlim = (-(l1+l2), (l1+l2)), ylim = (-(l1+l2), (l1+l2)))
ax.grid()

ln, = ax.plot([], [], "o-", lw=1.5, color="blue")

trail1, = ax.plot([], [], ",-", lw=1.3, color="green", alpha = 0.7)
trail2, = ax.plot([], [], ",-", lw=1.3, color="red", alpha = 0.7)

def init():
    ln.set_data([],[])
    return ln,

def animate(i):
    x = [0, x1[i], x2[i]]
    y = [0, y1[i], y2[i]]

    if i < N:
        k = 0
    else:
        k = i-N
    xt1 = [x2[k:i]]
    yt1 = [y2[k:i]]
    xt2 = [x1[k:i]]
    yt2 = [y1[k:i]]

    ln.set_data(x, y)
    trail1.set_data(xt1, yt1)
    trail2.set_data(xt2, yt2)
    return ln, trail1, trail2
    
anim = animation.FuncAnimation( fig, animate, np.arange(1, len(y)),
                                interval = 25, blit=True, init_func=init,
                                )

plt.show()
