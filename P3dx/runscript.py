import os
import sys

reps = int(sys.argv[1])
exp = sys.argv[2]

currentpath = os.getcwd()
os.system('mkdir '+exp)
os.chdir(currentpath+'/'+exp)
currentpath = os.getcwd()

for k in range(reps):
    print(k)
    os.system('time python ../evolve_ffann_invpend.py '+str(k)+' &')
