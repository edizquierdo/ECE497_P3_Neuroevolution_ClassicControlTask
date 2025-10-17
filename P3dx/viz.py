import numpy as np
import matplotlib.pyplot as plt
import sys

reps = int(sys.argv[1])
exp = sys.argv[2]

# Params taken directly from evolve (note: not the best way to do this)
layers = np.array([3,10,1])
genesize = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) 
generations = 100
trials_theta_exam = 50
trials_thetadot_exam = 50
theta_range_exam = np.linspace(-np.pi, np.pi, num=trials_theta_exam)
thetadot_range_exam = np.linspace(-1.0, 1.0, num=trials_thetadot_exam)
total_trials_exam = trials_theta_exam*trials_thetadot_exam
duration = 10
stepsize = 0.05
time = np.arange(0.0,duration,stepsize)

# Data structures
best = np.zeros((reps,generations))
theta = np.zeros((reps,total_trials_exam,len(time)))
fitmap = np.zeros((reps,len(theta_range_exam),len(thetadot_range_exam)))
bi = np.zeros((reps,genesize))

# Load data
for i in range(reps): 
    best[i] = np.load(exp+"/bestfit"+str(i)+".npy")
    theta[i] = np.load(exp+"/theta"+str(i)+".npy")
    fitmap[i] = np.load(exp+"/fitmap"+str(i)+".npy")
    bi[i] = np.load(exp+"/bestgenotype"+str(i)+".npy")

# Viz best fitness 
plt.plot(best.T)

# Figure out which one got the highest final best score (and plot it in black)
bestofbest_ind = -1
bestofbest_fit = -999999
bestfm_ind = -1
bestfm_fit = -999999
for i in range(reps): 
    print(i,np.mean(fitmap[i]),best[i][-1])
    if np.mean(fitmap[i]) > bestfm_fit:
        bestfm_fit = np.mean(fitmap[i])
        bestfm_ind = i 
    if best[i][-1] > bestofbest_fit:
        bestofbest_fit = best[i][-1]
        bestofbest_ind = i 

plt.plot(best[bestfm_ind],'k')
print(bestfm_ind)
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.show()

# Only for the best of the best, show their behavior: 
# Viz angle over time
plt.plot(theta[bestfm_ind].T)
plt.xlabel("Time")
plt.ylabel("Angle")
plt.show()

# Viz performance across different starting conditions
plt.imshow(fitmap[bestfm_ind])
plt.xlabel("Angle")
plt.ylabel("Angular acceleration")
plt.colorbar()
plt.show()





