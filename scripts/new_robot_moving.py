
from visualizer.visualizer import ODE
from pybullet_env.BasicEnvironment import SoftRobotBasicEnvironment

import time
import numpy as np

if __name__ == '__main__':

 
    env = SoftRobotBasicEnvironment()


    base_pos = np.array([0, -0.0, 0.06])
    base_orin = np.array([0.0,0,0])
    moving_speed = 0.03
    state = 0
    ts = 0.01
    
    for i in range(1000):
        # env.capture_image()
        # env.in_hand_camera_capture_image()
        
        t = time.time()
   

            
        base_pos[1] += ts*moving_speed
        base_pos[0] += ts*moving_speed
        
        base_orin[0] += ts*moving_speed*0
        base_orin[1] += ts*moving_speed*0
        base_orin[2] += ts*moving_speed*7
        
        
        
        # xc = env.move_robot(action=q,base_pos = base_pos)
        xc = env.move_robot_ori(action=np.array([0,0,0]),base_pos = base_pos, base_orin = base_orin)
            