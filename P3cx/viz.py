import numpy as np
import matplotlib.pyplot as plt
import env as bt

# Load data from files
a_x = np.load('a_x.npy')
a_y = np.load('a_y.npy')
l_x = np.load('l_x.npy')
l_y = np.load('l_y.npy')
d = np.load('d.npy')

# Visualize the agent's trajectory and color code it using time
#plt.scatter(a_x,a_y,s=0.5,c=range(len(a_x)),cmap="plasma")
plt.plot(a_x.T,a_y.T)
plt.plot(0.0,0.0,"k^")      # Visualize starting position of the agent with a black triangle
plt.plot(l_x,l_y,"ko")   # Visualize position of the light source with a black circle
plt.xlabel("x")
plt.xlabel("y")
plt.show()

# Figure of the distance over time for this same simulation
plt.imshow(d)
plt.xlabel("Repetitions")
plt.ylabel("Angle")
plt.colorbar()
plt.show()




