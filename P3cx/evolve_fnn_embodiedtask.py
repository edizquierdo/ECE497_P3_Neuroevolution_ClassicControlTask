import numpy as np
import matplotlib.pyplot as plt
import fnn 
import eas as ea 
import env

# Parameters of the task
duration = 1000  
distance = 5 
reps = 4

# Parameters of the neural network
layers = [2,4,2]

# Parameters of the evolutionary algorithm
genesize = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) 
print("Number of parameters: ",genesize)
popsize = 100
recombProb = 0.5
mutatProb = 0.05
generations = 10
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
        
        fitness += agent.distance(light) 
    
    return -fitness/reps

# Evolve
ga = ea.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations)
ga.run()
ga.showFitness()
np.save("best.npy",ga.bestind)
