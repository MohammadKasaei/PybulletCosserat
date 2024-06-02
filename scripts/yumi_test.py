
from environment_Yumi.yumiEnvSpatula import yumiEnvSpatula

from pybullet_env.BasicEnvironment import SoftRobotBasicEnvironment

import numpy as np

if __name__ == '__main__':

    env = yumiEnvSpatula()

    soft_robot_left = SoftRobotBasicEnvironment(moving_base=True,p = env._bullet,sphere_radius=0.01,_number_of_segment=1)
    soft_robot_right = SoftRobotBasicEnvironment(moving_base=True,p = env._bullet,head_color= [0,0.75,0,1], sphere_radius=0.01,_number_of_segment=1)
    
    t = 0
    dt = 0.01
    cam_pos = np.array([0,0,0])
    state = 0
    while True:    
        t += dt
        if state == 0 : # go home
            # env.go_home()
            
            pos_l = np.array([
                0.4 + 0.0 * np.sin(0.1*np.pi*t),
                0.2 + 0.0 * np.sin(0.1*np.pi*t),
                0.4 - 0.0 * np.sin(0.1*np.pi*t)
            ])
            ori_l = np.array([
                np.pi/10 + 0.0* np.sin(0.2*np.pi*t),
                np.pi - 0.0 * np.sin(0.02*np.pi*t),
                -np.pi/2 * np.sin(0.02*np.pi*t),
            ])
            
            pos_r = np.array([
                0.4+ 0.0 * np.sin(0.1*np.pi*t),
                -0.2 + 0.0 * np.sin(0.1*np.pi*t),
                0.4 + 0.0 * np.sin(0.2*np.pi*t)
            ])
            ori_r = np.array([
                -np.pi/10 + 0 * np.sin(0.2*np.pi*t),
                np.pi - 0.0 * np.sin(0.02*np.pi*t),
                -np.pi/2 * np.sin(0.02*np.pi*t),
            ])
            
            ori_l = env._bullet.getQuaternionFromEuler(ori_l)        
            xd  = np.array([0.5,0.4,0.6])
            
            pose_l = [pos_l,ori_l]        
            env.move_left_arm(traget_pose=pose_l)
            
            
            
            ori_r = env._bullet.getQuaternionFromEuler(ori_r)
            pose_r = [pos_r,ori_r]
            env.move_right_arm(traget_pose=pose_r)

        sf_left_seg1_cable_1   = .003*np.sin(0.5*np.pi*t)
        sf_left_seg1_cable_2   = 0.01+.005*np.sin(0.5*np.pi*t)
        sf_left_seg2_cable_1   = 0.005 + .00*np.sin(0.5*np.pi*t+1)
        sf_left_seg2_cable_2   = 0.005+.003*np.sin(0.5*np.pi*t+1)
        sf_left_seg3_cable_0   = .02*np.sin(0.5*np.pi*t)
        sf_left_seg3_cable_1   = .01*np.sin(0.5*np.pi*t+2)
        sf_left_seg3_cable_2   = .02*np.sin(0.5*np.pi*t+2)
        sf_left_gripper_pos    = np.abs(np.sin(np.pi*t))
                
        p0,o0 = env.get_left_ee_state()
        p0,o0 = env._bullet.multiplyTransforms(p0, o0, [0.0, 0.0,0.06], [0,0,0,1])
        angle = np.pi/2  
        rotation_quaternion = env._bullet.getQuaternionFromEuler([angle, 0, angle])
        
        new_pos, new_ori = env._bullet.multiplyTransforms(p0, o0, [0,0,0], rotation_quaternion)
        base_orin = env._bullet.getEulerFromQuaternion(new_ori)

        soft_robot_left.move_robot_ori(action=np.array([0.0, sf_left_seg1_cable_1, sf_left_seg1_cable_2, 
                                                    0.0, sf_left_seg2_cable_1, sf_left_seg2_cable_2,
                                                    sf_left_seg3_cable_0, sf_left_seg3_cable_1, sf_left_seg3_cable_2]),
                                base_pos = new_pos, base_orin = base_orin)
        
        
        
        sf_right_seg1_cable_1   = .003*np.sin(0.5*np.pi*t)
        sf_right_seg1_cable_2   = 0.01+.005*np.sin(0.5*np.pi*t)
        sf_right_seg2_cable_1   = 0.005 + .00*np.sin(0.5*np.pi*t+1)
        sf_right_seg2_cable_2   = 0.005+.003*np.sin(0.5*np.pi*t+1)
        sf_right_seg3_cable_0   = .02*np.sin(0.5*np.pi*t)
        sf_right_seg3_cable_1   = .01*np.sin(0.5*np.pi*t+2)
        sf_right_seg3_cable_2   = .02*np.sin(0.5*np.pi*t+2)
        sf_right_gripper_pos    = np.abs(np.sin(np.pi*t))
                
        p0,o0 = env.get_right_ee_state()
        p0,o0 = env._bullet.multiplyTransforms(p0, o0, [0.0, -0.00,0.06], [0,0,0,1])
        angle = np.pi/2  
        rotation_quaternion = env._bullet.getQuaternionFromEuler([angle, 0, angle])
        
        new_pos, new_ori = env._bullet.multiplyTransforms(p0, o0, [0,0,0], rotation_quaternion)
        base_orin = env._bullet.getEulerFromQuaternion(new_ori)

        soft_robot_right.move_robot_ori(action=np.array([0.0, sf_right_seg1_cable_1, sf_right_seg1_cable_2, 
                                                    0.0, sf_right_seg2_cable_1, sf_right_seg2_cable_2,
                                                    sf_right_seg3_cable_0, sf_right_seg3_cable_1, sf_right_seg3_cable_2]),
                                base_pos = new_pos, base_orin = base_orin)
        
        
        
    
    state = 0 
    while (True):


        if state == 0 : # go home
            # env.go_home()
            ori = env._bullet.getQuaternionFromEuler([0,np.pi,0])        
            xd  = np.array([0.5,0.4,0.6])
            pose_l = [xd,ori]        
            env.move_left_arm(traget_pose=pose_l)
            
            
            xd  = [0.5,-0.4,0.6]
            pose_r = [xd,ori]
            env.move_right_arm(traget_pose=pose_r)

            env.wait(10)