import numpy as np
import time

from environment.BasicEnvironment import BasicEnvironment
from pybullet_env.BasicEnvironment import SoftRobotBasicEnvironment



def Jac(f, q, dq=np.array((1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4))):
    
    fx0 = f(q[0],q[1],q[2])[0]
    n   = len(dq)
    m   = len(fx0)
    jac = np.zeros((n, m))
    for j in range(n):  # through rows 
        if (j==0):
            Dq = np.array((dq[0]/2.0,0,0, 0,0,0))
            jac [j,:] = (f(q[0]+Dq,q[1],q[2])[0] - f(q[0]-Dq,q[1],q[2])[0])/dq[j]
        elif (j==1):
            Dq = np.array((0,dq[1]/2.0,0 ,0,0,0))            
            jac [j,:] = (f(q[0]+Dq,q[1],q[2])[0] - f(q[0]-Dq,q[1],q[2])[0])/dq[j]
        elif (j==2):
            Dq = np.array((0,0,dq[2]/2.0,0,0,0))
            jac [j,:] = (f(q[0]+Dq,q[1],q[2])[0] - f(q[0]-Dq,q[1],q[2])[0])/dq[j]
        if (j==3):
            Dq = np.array((0,0,0, dq[3]/2.0,0,0))
            jac [j,:] = (f(q[0]+Dq,q[1],q[2])[0] - f(q[0]-Dq,q[1],q[2])[0])/dq[j]
        elif (j==4):
            Dq = np.array((0,0,0,0,dq[4]/2.0,0))            
            jac [j,:] = (f(q[0]+Dq,q[1],q[2])[0] - f(q[0]-Dq,q[1],q[2])[0])/dq[j]
        elif (j==5):
            Dq = np.array((0,0,0,0,0,dq[5]/2.0))
            jac [j,:] = (f(q[0]+Dq,q[1],q[2])[0] - f(q[0]-Dq,q[1],q[2])[0])/dq[j]
            
            
        elif (j==6):
            Dq = np.array((dq[6]/2.0,0,0))
            jac [j,:] = (f(q[0],q[1]+Dq,q[2])[0] - f(q[0],q[1]-Dq,q[2])[0])/dq[j]
        elif (j==7):
            Dq = np.array((0,dq[7]/2.0,0))
            jac [j,:] = (f(q[0],q[1]+Dq,q[2])[0] - f(q[0],q[1]-Dq,q[2])[0])/dq[j]
        elif (j==8):
            Dq = np.array((0,0, dq[8]/2.0))
            jac [j,:] = (f(q[0],q[1]+Dq,q[2])[0] - f(q[0],q[1]-Dq,q[2])[0])/dq[j]
            
    return jac    


def get_ref(gt,traj_name='Circle'):
    
        if traj_name == 'Rose':
            k = 4
            T  = 20
            w  = 2*np.pi/T
            a = 0.2
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
            T  = 20
            w  = 2*np.pi/T
            radius = 0.2
            xd = (x0 + np.array((radius*np.sin(w*(gt)),radius*np.cos(w*(gt)),-0.00*gt)))
            xd_dot = np.array((radius*w*np.cos(w*(gt)),-radius*w*np.sin(w*(gt)),-0.00))
            # xd = (x0 + np.array((0.00*gt,radius*np.sin(w*(gt)),radius*np.cos(w*(gt)))))
            # xd_dot = np.array((0.00,radius*w*np.cos(w*(gt)),-radius*w*np.sin(w*(gt))))
            
        elif traj_name=='Helix':
            T  = 20
            w  = 2*np.pi/T
            radius = 0.2
            xd = (x0 + np.array((radius*np.sin(w*(gt)),radius*np.cos(w*(gt)),-0.005*gt)))
            xd_dot = ( np.array((radius*w*np.cos(w*(gt)),-radius*w*np.sin(w*(gt)),-0.005)))
        elif traj_name=='Eight_Figure':
            T  = 10
            A  = 0.2
            w  = 2*np.pi/T
            xd = x0 + np.array((A*np.sin(w*gt) , A*np.sin((w/2)*gt),0.))
            xd_dot = np.array((A*w*np.cos(w*gt),A*w/2*np.cos(w/2*gt),0.00))
        elif traj_name=='Moving_Eight_Figure':
            T  = 15
            A  = 0.15
            w  = 2*np.pi/T
            xd = np.array(x0+(A*np.sin(w*gt) , A*np.sin((w/2)*gt),0.002*gt))
            xd_dot = np.array((A*w*np.cos(w*gt),A*w/2*np.cos(w/2*gt),0.002))
        elif traj_name=='Square':        
            T  = 5 #12.5*2
            tt = gt % (4*T)
            scale = 20

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
        elif traj_name=='Helictical_square':        
            T  = 5 #12.5*2
            tt = gt % (4*T)
            scale = 20

            if (tt<T):
                xd = (x0 + scale*np.array((-0.01+(0.02/T)*tt,0.01,-0.0002*gt)))
                xd_dot = scale*np.array(((0.02/T),0,-0.0002))
            elif (tt<2*T):
                xd = (x0 + scale*np.array((0.01,0.01-((0.02/T)*(tt-T)),-0.0002*gt)))
                xd_dot = scale*np.array((0,-(0.02/T),-0.0002))
            elif (tt<3*T):
                xd = (x0 + scale*np.array((0.01-((0.02/T)*(tt-(2*T))),-0.01,-0.0002*gt)))
                xd_dot = scale*np.array((-(0.02/T),0,-0.0002))
            elif (tt<4*T):
                xd = (x0 + scale*np.array((-0.01,-0.01+((0.02/T)*(tt-(3*T))),-0.0002*gt)))
                xd_dot = scale*np.array((0,+(0.02/T),-0.0002))
            else:
                # t0 = time.time()+5
                gt = 0
              
        elif traj_name=='Triangle':        
            T  = 12.5 *2
            tt = gt % (4*T)
            scale = 10
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


if __name__ == "__main__":
    
    saveLog = True
    
    env = BasicEnvironment()
    env.move_arm(target_pos= np.array([0.2,0.,0.35]), target_ori=[np.pi/2,np.pi/2,0],duration=0.01)
    env.wait(1)
    soft_robot = SoftRobotBasicEnvironment(bullet= env._pybullet,number_of_segment=2)
    env._pybullet.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0.5,0,0.5])
 
    t = 0
    dt = 0.01
    tf = 40
    ts = env._simulationStepTime
    # traj_name = 'Square'
    traj_name = 'Circle'
    # traj_name = 'Eight_Figure'
    # traj_name = 'Helictical_square'
    
    gt = 0.0
    
    q = np.array([0.0,0.0,0,0,0,0,0,0,0])    

    K = 0.35*np.diag((5.45, 5.45, 5.45))
    tp = time.time()
    t0 = tp
    ref = None
    
    pos = np.array([0.5 ,0.0 ,0.5])
    ori = np.array([np.pi/2,np.pi/2,0])
    env.move_arm (target_pos= pos, target_ori=ori)
    env._dummy_sim_step(100)
    p0,o0 = env.get_ee_state()
    p0,o0 = env._pybullet.multiplyTransforms(p0, o0, [0.025,-0.0,-0.0], [0,0,0,1])
    angle = -np.pi/2  # 90 degrees in radians
    rotation_quaternion = env._pybullet.getQuaternionFromEuler([0, 0, angle])
    new_pos, new_ori = env._pybullet.multiplyTransforms(p0, o0, [0,0,0], rotation_quaternion)
    base_orin = env._pybullet.getEulerFromQuaternion(new_ori)
    
    sf_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    shape, ode_sol = soft_robot.move_robot_ori(action=sf_action, base_pos = new_pos, base_orin = base_orin,camera_marker=False)
    xc = shape[-1][:3]
    x0 = np.copy(xc)
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    logFname = f"scripts/logs/log_{traj_name}" + timestr+".dat"
    logState = np.array([])
    
    prevPose = x0
    
    # plot refrence trajectory 
    for i in range(int(tf/(ts*20))):
        gt += (ts*20)
        xd, xd_dot = get_ref(gt,traj_name)
        env._pybullet.addUserDebugLine(prevPose, xd, [0, 0, 0.3], 5, 0) 
        prevPose = xd
        
    prevPose = x0
    gt = 0.0
    for i in range(int(tf/ts)):
    
        t = time.time()
        dt = t - tp
        tp = t
        print(f"t:{gt:.1}")
        
        xd, xd_dot = get_ref(gt,traj_name)
       
        if ref is None:
            ref = np.copy(xd)
        else:
            ref = np.vstack((ref, xd))
   
        jac = Jac(soft_robot.calc_tip_pos,[sf_action,new_pos,base_orin])   
        err = xd-xc
        err = err.clip(-0.1,0.1)
        qdot = jac @ (xd_dot + np.squeeze((K@(err)).T))
        q += (qdot * ts)
        
        
        sf_action = 0.9*sf_action + 0.1*(qdot[:6] * ts)
        sf_action[0] = 0
        sf_action[3] = 0
        sf_action = sf_action.clip(-0.02,0.02)
        
        pos += (qdot[-3:] * ts)
        
        env.move_arm (target_pos= pos, target_ori=ori)
        env._dummy_sim_step(10)

        p0,o0 = env.get_ee_state()
        p0,o0 = env._pybullet.multiplyTransforms(p0, o0, [0.025,-0.0,-0.0], [0,0,0,1])
        angle = -np.pi/2  # 90 degrees in radians
        rotation_quaternion = env._pybullet.getQuaternionFromEuler([0, 0, angle])
        new_pos, new_ori = env._pybullet.multiplyTransforms(p0, o0, [0,0,0], rotation_quaternion)
        base_orin = env._pybullet.getEulerFromQuaternion(new_ori)
        
        shape, ode_sol  = soft_robot.move_robot_ori(action=sf_action,
                                base_pos = new_pos, base_orin = base_orin,camera_marker=False)
        
        xc = shape[-1][:3]        
        
        if int(gt*100)%10 == 0:
            env._pybullet.addUserDebugLine(prevPose, xc, [1, 0, 0.3], 5, 0) 
            prevPose = xc

        if (saveLog):
            dummyLog = np.concatenate((np.array((gt, dt)), np.squeeze(xc), np.squeeze(xd), np.squeeze(
                xd_dot), np.squeeze(qdot), np.array((q[0], q[1], q[2]))))
            if logState.shape[0] == 0:
                logState = np.copy(dummyLog)
            else:
                logState = np.vstack((logState, dummyLog))

        gt += ts
        
    if (saveLog):
        with open(logFname, "w") as txt:  # creates empty file with header
            txt.write("#l,ux,uy,x,y,z\n")

        np.savetxt(logFname, logState, fmt='%.5f')
        print(f"log file has been saved: {logFname}")
    