import numpy as np

class Point2D:

    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def align(self, other):
        self.x = other.x
        self.y = other.y        

    def move(self, magnitude, angle):
        self.x += magnitude * np.cos(angle)
        self.y += magnitude * np.sin(angle)

    def dist(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def get(self):
        return [self.x, self.y]

class Light:
    
    def __init__(self, distance, angle):
        self.pos = Point2D()
        self.pos.move(distance, angle)

class Braitenberg:

    def __init__(self, neuralnetwork):
        self.pos = Point2D()                                    # agent's x and y position, starts in middle of world
        self.sensors = [0.0, 0.0]                               # left and right sensor values
        self.motors = [0.0, 0.0]                                # left and right motor values
        self.orientation = 0.0 #np.random.random()*2*np.pi      # agent's orientation, starts at random
        self.velocity = 0.01                                    # agent's velocity, starts at 0
        self.radius = 1.0                                       # the size/radius of the vehicle
        self.angleoffset = np.pi/2                              # angle offset for the placement of sensors 
        self.rs_pos = Point2D()
        self.ls_pos = Point2D()
        self.update_sensor_pos()
        self.controller = neuralnetwork
        
    def update_sensor_pos(self):
        self.rs_pos.align(self.pos)  
        self.rs_pos.move(self.radius, self.orientation + self.angleoffset)
        self.ls_pos.align(self.pos)
        self.ls_pos.move(self.radius, self.orientation - self.angleoffset) 

    def sense(self, light):
        # Calculate the distance of the light for each of the sensors
        self.sensors[0] = np.clip(1 - self.rs_pos.dist(light.pos)/10, 0, 1)     # Right sensor
        self.sensors[1] = np.clip(1 - self.ls_pos.dist(light.pos)/10, 0, 1)     # Left sensor
        
    def move(self):
        # Make sure motors are within bounds
        self.motors = np.clip(self.motors, 0, 1)

        # Update the orientation and velocity of the vehicle based on the left and right motors
        self.orientation += ((self.motors[1] - self.motors[0])/10) + np.random.normal(0,0.01) 
        self.velocity = ((self.motors[1] + self.motors[0])/2)/100

        # Update position of the agent
        self.pos.move(self.velocity, self.orientation)

        # Update position of the sensors
        self.update_sensor_pos() 

    def think(self):
        self.motors = self.controller.forward(self.sensors)[0]
        #print(self.sensors, self.motors)
        # self.motors[0] = self.sensors[1]
        # self.motors[1] = self.sensors[0]
        
    def distance(self,light):
        return self.pos.dist(light.pos)

