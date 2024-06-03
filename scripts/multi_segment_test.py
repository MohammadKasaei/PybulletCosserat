

from pybullet_env.BasicEnvironment import SoftRobotBasicEnvironment

import numpy as np

if __name__ == '__main__':

    env = SoftRobotBasicEnvironment(body_sphere_radius=0.02,number_of_segment=5)
    base_link_shape = env._pybullet.createVisualShape(env._pybullet.GEOM_BOX, halfExtents=[0.05, 0.05, 0.03], rgbaColor=[0.6, 0.6, 0.6, 1])
    base_link_pos, base_link_ori = env._pybullet.multiplyTransforms([0,0,0.5], [0,0,0,1], [0,-0.0,0], [0,0,0,1])
    base_link_id    = env._pybullet.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=base_link_shape,
                                                        baseVisualShapeIndex=base_link_shape,
                                                        basePosition= base_link_pos , baseOrientation=base_link_ori)
       
       
    
    # for i in range(2):   
    #     color = np.ones([1,4]).squeeze()
    #     color[:3] = np.random.rand(1,3).squeeze()
    #     obj_shape = env._pybullet.createVisualShape(env._pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=color)
    #     obj_pos, obj_ori = env._pybullet.multiplyTransforms([0.005*np.random.randint(-100,100),0.005*np.random.randint(-100,100),0.05], [0,0,0,1], [0,-0.0,0], [0,0,0,1])
    #     obj_id    = env._pybullet.createMultiBody(baseMass=0.00, baseCollisionShapeIndex=base_link_shape,
    #                                                         baseVisualShapeIndex=obj_shape,
    #                                                         basePosition= obj_pos , baseOrientation=obj_ori)
          
    t = 0
    dt = 0.01
    cam_pos = np.array([0,0,0])
    state = 0
    while True:    
        t += dt
       

        sf_left_seg1_cable_1   = 0.005*np.sin(0.5*np.pi*t)
        sf_left_seg1_cable_2   = 0.005*np.sin(0.5*np.pi*t)
        sf_left_seg2_cable_1   = .004*np.sin(0.4*np.pi*t)
        sf_left_seg2_cable_2   = .003*np.sin(0.4*np.pi*t)
        sf_left_seg3_cable_0   = .001*np.sin(0.3*np.pi*t)
        sf_left_seg3_cable_1   = .001*np.sin(0.3*np.pi*t)
        sf_left_seg3_cable_2   = .001*np.sin(0.3*np.pi*t)
        sf_left_seg4_cable_1   = .005*np.sin(0.2*np.pi*t)
        sf_left_seg4_cable_2   = .005*np.sin(0.2*np.pi*t)
        sf_left_seg5_cable_1   = .005*np.sin(0.1*np.pi*t)
        sf_left_seg5_cable_2   = .005*np.sin(0.1*np.pi*t)
        
        
        
        sf_left_gripper_pos    = np.abs(np.sin(0.1*np.pi*t))
         
        shape, ode_sol = env.move_robot_ori(action=np.array([0.0, sf_left_seg1_cable_1, sf_left_seg1_cable_2, 
                                            0.0, sf_left_seg2_cable_1, sf_left_seg2_cable_2,
                                            sf_left_seg3_cable_0, sf_left_seg3_cable_1, sf_left_seg3_cable_2,
                                            0.0, sf_left_seg4_cable_1, sf_left_seg4_cable_2, 
                                            0.0, sf_left_seg5_cable_1, sf_left_seg5_cable_2]),
                                    base_pos=[0,0,0.5],base_orin = np.array([-np.pi/2,0,0]), camera_marker=False)
        
        marker_pos, marker_ori = env._pybullet.multiplyTransforms(shape[-1][:3], shape[-1][3:], [0.0,0.0,0.0], [0,0,0,1])

        env._set_marker(marker_pos,marker_ori)
        env.set_grasp_width(sf_left_gripper_pos)
        