from pybullet_env.BasicEnvironment_TwoSegments import SoftRobotBasicEnvironment_TwoSegments
import numpy as np
import time



def Jac(f, q, dq=np.array((1e-4,1e-4,1e-4,1e-4,1e-4,1e-4))):
    
    fx0 = f(q)
    n   = len(q)
    m   = len(fx0)
    jac = np.zeros((n, m))
    for j in range(n):  # through rows 
        if (j==0):
            Dq = np.array((dq[0]/2.0,0,0,0,0,0))
        elif (j==1):
            Dq = np.array((0,dq[1]/2.0,0,0,0,0))
        elif (j==2):
            Dq = np.array((0,0,dq[2]/2.0,0,0,0))
        elif (j==3):
            Dq = np.array((0,0,0,dq[2]/2.0,0,0))
        elif (j==4):
            Dq = np.array((0,0,0,0,dq[2]/2.0,0))
        elif (j==5):
            Dq = np.array((0,0,0,0,0, dq[2]/2.0))
            
        jac [j,:] = (f(q+Dq) - f(q-Dq))/dq[j]
    return jac    


def get_ref(gt,traj_name='Circle'):
    
        if traj_name == 'Rose':
            k = 4
            T  = 200
            w  = 2*np.pi/T
            a = 0.025
            r  = a * np.cos(k*w*gt)
            xd = (x0 + np.array((r*np.cos(w*gt),r*np.sin(w*gt),0.00*gt)))
            xd_dot = np.array((-r*w*np.sin(w*gt),r*w*np.cos(w*gt),0.00*gt))
        elif traj_name == 'Limacon':
            T  = 100
            w  = 2*np.pi/T
            radius = 0.02
            radius2 = 0.03
            shift = -0.02
            xd = (x0 + np.array(((shift+(radius+radius2*np.cos(w*gt))*np.cos(w*gt)),(radius+radius2*np.cos(w*gt))*np.sin(w*gt),0.00*gt)))
            xd_dot = np.array((radius*(-w*np.sin(w*(gt)-0.5*w*np.sin(w/2*(gt)))),radius*(w*np.cos(w*(gt)-0.5*radius2*np.cos(w/2*gt))),0.00))                            
        elif traj_name=='Circle':
            T  = 50*2
            w  = 2*np.pi/T
            radius = 0.02
            xd = (x0 + np.array((radius*np.sin(w*(gt)),radius*np.cos(w*(gt)),0.00*gt)))
            xd_dot = np.array((radius*w*np.cos(w*(gt)),-radius*w*np.sin(w*(gt)),0.00))
        elif traj_name=='Helix':
            T  = 50
            w  = 2*np.pi/T
            radius = 0.04
            xd = (x0 + np.array((radius*np.sin(w*(gt)),radius*np.cos(w*(gt)),0.0001*gt)))
            xd_dot = ( np.array((radius*w*np.cos(w*(gt)),-radius*w*np.sin(w*(gt)),0.0001)))
        elif traj_name=='Eight_Figure':
            T  = 25*2
            A  = 0.02
            w  = 2*np.pi/T
            xd = np.array((A*np.sin(w*gt) , A*np.sin((w/2)*gt),0.1))
            xd_dot = np.array((A*w*np.cos(w*gt),A*w/2*np.cos(w/2*gt),0.00))
        elif traj_name=='Moving_Eight_Figure':
            T  = 25*2
            A  = 0.03
            w  = 2*np.pi/T
            xd = np.array(x0+(A*np.sin(w*gt) , A*np.sin((w/2)*gt),0.0002*gt))
            xd_dot = np.array((A*w*np.cos(w*gt),A*w/2*np.cos(w/2*gt),0.0002))
        elif traj_name=='Square':        
            T  = 12.5*2
            tt = gt % (4*T)
            scale = 3

            if (tt<T):
                xd = (x0 + scale*np.array((-0.01+(0.02/T)*tt,0.01,0.0)))
                xd_dot = scale*np.array(((0.02/T),0,0))
            elif (tt<2*T):
                xd = (x0 + scale*np.array((0.01,0.01-((0.02/T)*(tt-T)),0.0)))
                xd_dot = scale*np.array((0,-(0.02/T),0))
            elif (tt<3*T):
                xd = (x0 + scale*np.array((0.01-((0.02/T)*(tt-(2*T))),-0.01,0.0)))
                xd_dot = scale*np.array((-(0.02/T),0,0))
            elif (tt<4*T):
                xd = (x0 + scale*np.array((-0.01,-0.01+((0.02/T)*(tt-(3*T))),0.0)))
                xd_dot = scale*np.array((0,+(0.02/T),0))
            else:
                # t0 = time.time()+5
                gt = 0
        elif traj_name=='Moveing_Square':        
            T  = 10.0
            tt = gt % (4*T)
            if (tt<T):
                xd = (x0 + 2*np.array((-0.01+(0.02/T)*tt,0.01,-0.02+0.0005*gt)))
                xd_dot = 2*np.array(((0.02/T),0,0.0005))
            elif (tt<2*T):
                xd = (x0 + 2*np.array((0.01,0.01-((0.02/T)*(tt-T)),-0.02+0.0005*gt)))
                xd_dot = 2*np.array((0,-(0.02/T),0.0005))
            elif (tt<3*T):
                xd = (x0 + 2*np.array((0.01-((0.02/T)*(tt-(2*T))),-0.01,-0.02+0.0005*gt)))
                xd_dot = 2*np.array((-(0.02/T),0,0.0005))
            elif (tt<4*T):
                xd = (x0 + 2*np.array((-0.01,-0.01+((0.02/T)*(tt-(3*T))),-0.02+0.0005*gt)))
                xd_dot = 2*np.array((0,+(0.02/T),0.0005))
              
        elif traj_name=='Triangle':        
            T  = 12.5 *2
            tt = gt % (4*T)
            scale = 2
            if (tt<T):
                xd = (x0 + scale*np.array((-0.01+(0.02/T)*tt,-0.01+(0.02/T)*tt,0.0)))
                xd_dot = scale*np.array(((0.02/T),(0.02/T),0))
            elif (tt<2*T):
                xd = (x0 + scale*np.array((0.01+(0.02/T)*(tt-(T)),0.01-((0.02/T)*(tt-(T))),0.0)))
                xd_dot = scale*np.array(((0.02/T),-(0.02/T),0))
            elif (tt<4*T):
                xd = (x0 + scale*np.array((0.03-((0.02/T)*(tt-(2*T))),-0.01,0.0)))
                xd_dot = scale*np.array((-(0.02/T),0,0))
            else:
                # t0 = time.time()+5
                gt = 0
        else: # circle
            T  = 50*2
            w  = 2*np.pi/T
            radius = 0.02
            xd = (x0 + np.array((radius*np.sin(w*(gt)),radius*np.cos(w*(gt)),0.00*gt)))
            xd_dot = np.array((radius*w*np.cos(w*(gt)),-radius*w*np.sin(w*(gt)),0.00))
            
        return xd,xd_dot


if __name__ == '__main__':

    env = SoftRobotBasicEnvironment_TwoSegments(moving_base=False)
    # env.move_robot(np.array([0.0,0,0,0,0,0]))
    
    tf = 100
    ts = 0.05
    traj_name = 'Square'
    gt = 0
    
    q = np.array([0.0,0.0,0,0,0,0])    
    # J = Jac(env._move_robot_jac,q)    

    x0 = env.move_robot(q)[:3]+np.array([0,0,0.0])
    xc = env.move_robot(q)[:3]
    K = 1*np.diag((2.45, 2.45, 2.45))
    tp = time.time()
    t0 = tp
    ref = None
    
    for i in range(int(tf/ts)):
        t = time.time()
        dt = t - tp
        tp = t
        
        xd, xd_dot = get_ref(gt,traj_name)
       
        if ref is None:
            ref = np.copy(xd)
        else:
            ref = np.vstack((ref, xd))
   
        jac = Jac(env._move_robot_jac,q).T
        pseudo_inverse = np.linalg.pinv(jac)
        qdot = pseudo_inverse @ (xd_dot + np.squeeze((K@(xd-xc)).T))
        q += (qdot * ts)
        
        xc = env.move_robot(q)[:3]
        gt += ts
        # ee = env.move_robot(action=q)    
        
    sign = 1   
    env._set_marker(np.array([0.2,0,0]),radius=0.035,color=[0,1,0,1])
     
    while True:    
        for i in range(200):
            env.move_robot(np.array([0.0,sign*i/10000.0,0.0, 0, -sign*i/10000.0,0.0]))
            time.sleep(0.01)
            env._dummy_sim_step(100)
            if i % 10 == 0 :
                env.in_hand_camera_capture_image()    
        
            
        for i in range(200,0,-1):
            env.move_robot(np.array([0.0,sign*i/10000.0,0.0, 0, -sign*i/10000.0,0.0]))
            time.sleep(0.01)
            env._dummy_sim_step(100)
            if i % 10 == 0 :
                env.in_hand_camera_capture_image()    
        
        sign = -1 if sign==1 else 1
        
            
        
    
