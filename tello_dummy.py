import numpy as np
import matplotlib.pyplot as plt
import math
from tello import Tello as drone
#from tello_sim import Simulator

class Tello:
    def __init__(self):
        self.all_pos = []
        self.cur_pos = [0, 0, 0]
        self.all_pos += [self.cur_pos]
        self.last_command = 'stop'
        plt.ion()
        plt.title('Dummy Drone Path')
        self.ax = plt.axes(projection='3d')
        self.drone = drone()
        #Simulator drone
        # self.drone = Simulator()

    def send_command(self, command):
        print('Tello Received: {}'.format(command))

        if 'continue' != command or not 'stop' in command:
            self.drone.send_command(command)

            

    def send_command_droneless(self, command):
        print('Tello Received: {}'.format(command))
        if command=='takeoff':
            self.cur_pos[1] += 100
            self.all_pos += [self.cur_pos]
        elif 'go' in self.last_command:
            traveling_dir = [int(x) for x in self.last_command.split(' ')[1:-1]]
            traveling_mag = math.sqrt(sum([x*x for x in traveling_dir]))
            print(traveling_dir)
            self.cur_pos = [x+(y/traveling_mag) for (x, y) in zip(self.cur_pos, traveling_dir)]
            self.all_pos += [self.cur_pos]

        if 'continue' != command:
            self.last_command = command
        #self.update_plot()

    def update_plot(self):
        np_arr = np.array(self.all_pos)
        print(np_arr.shape)
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Z')
        self.ax.set_zlabel('Y')
        self.ax.plot3D(-np_arr[:,0], np_arr[:,2], np_arr[:,1])
        plt.draw()
        #plt.pause(0.00001)

    def plot(self):
        np_arr = np.array(self.all_pos)
        plt.ioff()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Z')
        self.ax.set_zlabel('Y')
        self.ax.plot3D(-np_arr[:,0], np_arr[:,2], np_arr[:,1])
        plt.show()
