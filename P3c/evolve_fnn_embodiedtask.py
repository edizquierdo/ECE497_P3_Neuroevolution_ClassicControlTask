##################################################################################
# Example script for evolving a feedforward neural network to solve XOR problem 
#
# Eduardo Izquierdo
# May 2024
##################################################################################

import numpy as np
import matplotlib.pyplot as plt
import fnn 
import eas as ea 
import env

# Parameters of the task
duration = 1000  
distance = 10 
reps = 4

# Parameters of the neural network
layers = [2,10,2]

# Parameters of the evolutionary algorithm
genesize = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) + (len(layers)-1)*6 
print("Number of parameters: ",genesize)
popsize = 10 #100
recombProb = 0.0
mutatProb = 0.05
generations = 50 #1000
demeSize = 5

def fitnessFunction(genotype):
    # Step 1: Create the neural network and set the parameters according to the genotype.
    controller = fnn.FNN(layers)
    controller.setParams(genotype)
    
    angle = 0.0
    fitness = 0
    for r in range(reps):
        # Step 2: Create the body and env. 
        agent = env.Braitenberg(controller)
        angle += np.pi/2
        light = env.Light(distance, angle) 

        # Step 3. Run simulation 
        aggdist = 0
        for t in range(duration):
            agent.sense(light)
            agent.think()
            agent.move()
            aggdist += agent.distance(light)
        
        fitness += np.clip(np.abs(distance - (aggdist/duration))/distance, 0, 1)
    
    return fitness/reps 

# Evolve
ga = ea.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations)
ga.run()
ga.showFitness()

# # Obtain best final solution and create a neural network with it
# avgfit, bestfit, bestind = ga.fitStats()
# a = fnn.FNN(layers)
# a.setParams(bestind)

# # Function to visualize 
# def viz(neuralnet, dataset, label):
#     X = np.linspace(-1.05, 1.05, 100)
#     Y = np.linspace(-1.05, 1.05, 100)
#     output = np.zeros((100,100))
#     i = 0
#     for x in X: 
#         j = 0
#         for y in Y: 
#             output[i,j] = neuralnet.forward([x,y])
#             j += 1
#         i += 1
#     plt.contourf(X,Y,output)
#     plt.colorbar()
#     plt.xlabel("x")
#     plt.ylabel("y")
#     for i in range(len(dataset)):
#         if label[i] == 1:
#             plt.plot(dataset[i][0],dataset[i][1],'wo')
#         else:
#             plt.plot(dataset[i][0],dataset[i][1],'wx')
#     plt.show()  

# # Visualize data
# viz(a, dataset, labels)
