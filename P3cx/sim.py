import numpy as np
import matplotlib.pyplot as plt
import env as bt
import fnn 

duration = 1000
distance = 5
repetitions = 20
angles = np.linspace(0,1,10,endpoint=False)*2*np.pi

# Parameters of the neural network
layers = [2,4,2]

def SimInd(genotype):

    controller = fnn.FNN(layers)
    controller.setParams(genotype)

    # Variables (store time-varying variables)
    xpos = np.zeros((repetitions*len(angles),duration))
    ypos = np.zeros((repetitions*len(angles),duration))
    dpos = np.zeros((len(angles),repetitions))
    lxps = np.zeros(repetitions*len(angles))
    lyps = np.zeros(repetitions*len(angles))
    i = 0
    for r in range(repetitions):
        a = 0
        for angle in angles: 

            # First, instantiate a vehicle and a light source
            agent = bt.Braitenberg(controller)
            light = bt.Light(distance,angle)

            # Run // Simulation 
            for t in range(duration):
                # Stepping // Updating
                agent.sense(light)
                agent.think()
                agent.move()
                # Keeping track of data
                xpos[i][t] = agent.pos.x
                ypos[i][t] = agent.pos.y
                
            dpos[a][r] = agent.distance(light)
            a += 1
            lxps[i] = light.pos.x
            lyps[i] = light.pos.y
            i += 1

    print("Fitness:",-np.mean(dpos))
    return xpos,ypos,lxps,lyps,dpos


# Simulate and save data in variables
genotype = np.load('best.npy')
print(genotype)
a_x, a_y, l_x, l_y, d = SimInd(genotype)

# Store variables in a file
np.save('a_x.npy',a_x)
np.save('a_y.npy',a_y)
np.save('d.npy',d)
np.save('l_x.npy',l_x)
np.save('l_y.npy',l_y)







