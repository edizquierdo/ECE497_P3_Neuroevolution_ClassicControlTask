import numpy as np
import math

class MountainCar:

    def __init__(self):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45   # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.goal_velocity = 0.0
        self.power = 0.0015
        self.position = 0.0
        self.velocity = 0.0
        
    def step(self, stepsize, action):
        force = min(max(action[0], self.min_action), self.max_action)
        self.velocity += force * self.power - 0.0025 * math.cos(3 * self.position)
        if (self.velocity > self.max_speed): self.velocity = self.max_speed
        if (self.velocity < -self.max_speed): self.velocity = -self.max_speed
        self.position += self.velocity
        if (self.position > self.max_position): self.position = self.max_position
        if (self.position < self.min_position): self.position = self.min_position
        if (self.position == self.min_position and self.velocity < 0): velocity = 0
        done = bool(self.position >= self.goal_position and self.velocity >= self.goal_velocity)
        if not done:
            return - 1.0 * stepsize, done
        else:
            return 0.0, done

    def state(self):
        return np.array([self.position, self.velocity])
