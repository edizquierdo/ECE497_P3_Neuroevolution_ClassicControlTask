import eas                  # Optimimizer
import fnn                  # Controller
import invpend              # Task

import numpy as np
import matplotlib.pyplot as plt
import sys

# Input from terminal 
number = int(sys.argv[1])           # Receive the number of the experiment for saving data

# NN Params
layers = np.array([3,10,1])
duration = 10
stepsize = 0.05
WeightRange = 15.0 
BiasRange = 15.0 
noisestd = 0.01 

# Fitness initialization ranges
trials_theta = 6
trials_thetadot = 6
total_trials = trials_theta*trials_thetadot
theta_range = np.linspace(-np.pi, np.pi, num=trials_theta)
thetadot_range = np.linspace(-1.0,1.0, num=trials_thetadot)
time = np.arange(0.0,duration,stepsize)

# EA Params
genesize = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) 
print("Number of parameters: ",genesize)
popsize = 100
recombProb = 0.5
mutatProb = 0.05
generations = 100
demeSize = 5

# Fitness function
def fitnessFunction(genotype):
    nn = fnn.FNN(layers)
    nn.setParams(genotype)
    nn.weightrange = WeightRange
    nn.biasrange = BiasRange
    body = invpend.InvPendulum()
    fit = 0.0
    for theta in theta_range:
        for theta_dot in thetadot_range:
            body.theta = theta
            body.theta_dot = theta_dot
            for t in time:
                outputs = nn.forward(body.state())
                f = body.step(stepsize, outputs[0] + np.random.normal(0.0,noisestd))
                fit += f
    return fit/(duration*total_trials)

# Evolve and visualize fitness over generations
ga = eas.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations)
ga.run()

# Get best evolved network and show its activity
af,bf,bi = ga.fitStats()

# Params for more general evaluation of the evolved solution
trials_theta_exam = 50
trials_thetadot_exam = 50
theta_range_exam = np.linspace(-np.pi, np.pi, num=trials_theta_exam)
thetadot_range_exam = np.linspace(-1.0, 1.0, num=trials_thetadot_exam)
total_trials_exam = trials_theta_exam*trials_thetadot_exam

def evaluate(genotype):     # repeat of fitness function but saving theta
    nn = fnn.FNN(layers)
    nn.setParams(genotype)
    nn.weightrange = WeightRange
    nn.biasrange = BiasRange
    body = invpend.InvPendulum()
    fit = 0.0
    theta_hist=np.zeros((total_trials_exam,len(time))) 
    rep = 0 
    kt = 0
    fitmap = np.zeros((len(theta_range_exam),len(thetadot_range_exam)))
    for theta in theta_range_exam:
        ktd = 0
        for theta_dot in thetadot_range_exam:
            k = 0 
            body.theta = theta
            body.theta_dot = theta_dot
            indfit = 0.0
            for t in time:
                outputs = nn.forward(body.state())
                f = body.step(stepsize, outputs[0])
                theta_hist[rep][k] = body.theta 
                indfit += f
                k += 1 
            fitmap[kt][ktd]=indfit/duration
            ktd += 1
            fit += indfit
            rep += 1 
        kt += 1
    print(fit/(duration*total_trials_exam))
    return theta_hist, fitmap

theta_hist, fitmap = evaluate(bi)

# Viz 
#ga.showFitness()
# plt.plot(theta_hist.T)
# plt.xlabel("Time")
# plt.ylabel("Angle")
# plt.show()

# plt.imshow(fitmap)
# plt.xlabel("Angle")
# plt.ylabel("Angular acceleration")
# plt.colorbar()
# plt.show()

# Save data
np.save("bestfit"+str(number)+".npy",ga.bestHistory)
np.save("avgfit"+str(number)+".npy",ga.avgHistory)
np.save("theta"+str(number)+".npy",theta_hist)
np.save("fitmap"+str(number)+".npy",fitmap)
np.save("bestgenotype"+str(number)+".npy",bi)







