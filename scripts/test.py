import pybullet as p
import pybullet_data as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from scripts.CPG import CPG
from pybullet_env.BasicEnvironment import SoftRobotBasicEnvironment


class MiniSpotEnv():
    def __init__(self) -> None:            
        self.physicsClient = p.connect(p.GUI)
        self._pybullet = p
        self._samplingTime = 0.005
        p.setTimeStep(self._samplingTime)

        p.setAdditionalSearchPath(pd.getDataPath())
        p.setGravity(0,0,-9.81)

        self.FloorId = p.loadURDF("plane.urdf",[0,0,-0.])        
        self.robotID = p.loadURDF("environment/urdf_models/spotmicro_proprio_v5/urdf/spotmicro_proprio_v5.urdf",[0,0,0.2])

        boundaries = p.getAABB(self.robotID,1)


        self.colorPalettes = {
            "lightOrange": [1.0, 0.82, 0.12, 1.0],
            "darkOrange": [1.0, 0.6, 0.0, 1.0],
            "darkGrey": [0.43, 0.43, 0.43, 1.0],
            "lightGrey": [0.65, 0.65, 0.65, 1.0],
        }


        p.resetVisualShapeData(
                    self.robotID, 0, 
                    rgbaColor=self.colorPalettes["darkOrange"])

        p.resetVisualShapeData(
                    self.robotID, 3, 
                    rgbaColor=self.colorPalettes["darkOrange"])



        #Lock joints in place
        numJoints = p.getNumJoints(self.robotID)
        for j in range(numJoints):
            p.setJointMotorControl2( bodyIndex = self.robotID, jointIndex = j, controlMode = p.POSITION_CONTROL, targetPosition = 0 )
            aabb = p.getAABB(self.robotID, j)
            aabbMin = np.array(aabb[0])
            aabbMax = np.array(aabb[1])
            print(aabbMax - aabbMin)
            
            # print(aabbMin)
            # print(aabbMax)
        
            info = p.getJointInfo(self.robotID, j)
            print (info)

        self.tsim = 0
        self.JointPositions = np.zeros(12)    
        
        self.reset(-0.23)

                            
    def get_ee_state(self):
        pose = p.getLinkState(self.robotID,0)[0:2]        
        return pose[0] , pose[1]
        
    def reset(self,zleg):
        stabilization_steps = 1000
        p.setTimeStep(self._samplingTime)
        for _ in range(stabilization_steps): 
            p.stepSimulation()
            pFL = np.array((0.0,0.0,zleg))
            pFR = np.array((0.0,-0.0,zleg))
            pBL = np.array((0.0,0.0,zleg))
            pBR = np.array((0.0,-0.0,zleg))
            FL = self.IK(pFL)
            FR = self.IK(pFR)
            BL = self.IK(pBL)
            BR = self.IK(pBR)
            self.JointPositions[0:3] = FL
            self.JointPositions[3:6] = FR
            self.JointPositions[6:9] = BL
            self.JointPositions[9:12]= BR
            self.applyMotorCommand()

    def IK(self,targetPosition):

        Lu = 0.118
        Ld = 0.113
        epsilon = 0.000001

        Lx2 =  (0*targetPosition[0]**2) + (targetPosition[2]**2)
        Ly2 =  (targetPosition[1]**2) + (targetPosition[2]**2)

        Lx = np.sqrt(Lx2)
        Ly = np.sqrt(Ly2)

        Lu2 = Lu ** 2
        Ld2 = Ld ** 2

        alpha = (Lu2+Ld2-Lx2)/((2*Lu*Ld)+epsilon)
        if alpha>1: 
            alpha =1 
        elif (alpha<-1):
            alpha=-1

        thetaKnee  = np.arccos(alpha) - np.pi
        
        beta = (Lu2+Lx2-Ld2)/(2*Lu*Lx+epsilon)
        if (beta>1):
            beta = 1
        elif (beta<-1):
            beta = -1


        if (targetPosition[0]>=0):
            thetaHipx  = np.arccos(beta) + np.arctan(np.abs(targetPosition[0]) / (Lx+epsilon))
        else:
            thetaHipx  = np.arccos(beta) - np.arctan(np.abs(targetPosition[0]) / (Lx+epsilon))
            

        thetaHipy  = np.arctan(targetPosition[1] / (Ly+epsilon))
        JointPoses = [thetaHipy,thetaHipx,thetaKnee]
        
        return JointPoses 

    def applyMotorCommand(self):
        for i in range(len(self.JointPositions)):
            p.setJointMotorControl2(bodyIndex=self.robotID, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                                targetPosition=self.JointPositions[i], force=20)


    def step(self,cpg):

        if cpg.gtime < 5:
            cpg.NewStepX_raw = 0.1
            cpg.NewStepY_raw = 0.0
            cpg.NewStepTheta_raw = 0
        elif cpg.gtime < 20:
            cpg.NewStepX_raw = 0.1
            cpg.NewStepY_raw = 0.0
            cpg.NewStepTheta_raw = 0*np.pi/8
        elif cpg.gtime < 25:
            cpg.NewStepX_raw = 0.1
            cpg.NewStepY_raw = 0.0
            cpg.NewStepTheta_raw = 0*np.pi/8
        elif cpg.gtime < 30:
            cpg.NewStepX_raw = 0.1
            cpg.NewStepY_raw = 0.0
            cpg.NewStepTheta_raw = 0*np.pi/8
        





        cpg.gtime = self.tsim
        cpg.apply_walk_command()
        cpg.updateOmniJoints_CPG()
        
        StepTheta = cpg.NewStepTheta_raw            

        pFL = np.array([-0.0 + cpg.LfootPosition[0], 0.00 + cpg.LfootPosition[1], 0.00 + cpg.LfootPosition[2]])
        pFR = np.array([-0.0 + cpg.RfootPosition[0], 0.00 + cpg.RfootPosition[1], 0.00 + cpg.RfootPosition[2]])

        pBL = np.array([-0.0 + cpg.RfootPosition[0],-0.00 + cpg.RfootPosition[1], 0.00 +cpg.RfootPosition[2]])
        pBR = np.array([-0.0 + cpg.LfootPosition[0],-0.00 + cpg.LfootPosition[1], 0.00 + cpg.LfootPosition[2]])


        pFL[0] = pFL[0]*np.cos(StepTheta)-pFL[1]*np.sin(StepTheta)
        pFL[1] = pFL[0]*np.sin(StepTheta)+pFL[1]*np.cos(StepTheta)

        pFR[0] = pFR[0]*np.cos(StepTheta)-pFR[1]*np.sin(StepTheta)
        pFR[1] = pFR[0]*np.sin(StepTheta)+pFR[1]*np.cos(StepTheta)

        pBL[0] = pBL[0]*np.cos(StepTheta)-pBL[1]*np.sin(StepTheta)
        pBL[1] = pBL[0]*np.sin(StepTheta)+pBL[1]*np.cos(StepTheta)

        pBR[0] = pBR[0]*np.cos(StepTheta)-pBR[1]*np.sin(StepTheta)
        pBR[1] = pBR[0]*np.sin(StepTheta)+pBR[1]*np.cos(StepTheta)
        
        
        FL = self.IK(pFL)
        FR = self.IK(pFR)
        BL = self.IK(pBL)
        BR = self.IK(pBR)
        self.JointPositions[0:3] = FR
        self.JointPositions[3:6] = FL
        self.JointPositions[6:9] = BR
        self.JointPositions[9:12]= BL



        self.applyMotorCommand()
        
        p.stepSimulation()
        self.tsim += self._samplingTime
        time.sleep(self._samplingTime)

        print ("Finished")


        



if __name__ == "__main__":
    env = MiniSpotEnv()
    cpg = CPG()
    cpg.NewStepX_raw = 0.0
    cpg.NewStepY_raw = 0.0
    cpg.NewStepTheta_raw = 0

    
    soft_robot_1 = SoftRobotBasicEnvironment(moving_base=True,p = env._pybullet)
    base_link_id = None
    t = 0
    dt = 0.01
    cam_pos = np.array([0,0,0])
    while True:    
        t += dt
        env.step(cpg)
    
        
        sf1_seg1_cable_1   = .005*np.sin(0.5*np.pi*t)
        sf1_seg1_cable_2   = 0.007+.005*np.sin(0.5*np.pi*t)
        sf1_seg2_cable_1   = 0.0 + .00*np.sin(0.5*np.pi*t+1)
        sf1_seg2_cable_2   = 0.01+.00*np.sin(0.5*np.pi*t+1)
        sf1_seg3_cable_0   = .01*np.sin(0.5*np.pi*t)
        sf1_seg3_cable_1   = .01*np.sin(0.5*np.pi*t+2)
        sf1_seg3_cable_2   = np.abs(.02*np.sin(0.5*np.pi*t+2))
        sf1_gripper_pos    = np.abs(np.sin(np.pi*t))
                
        p0,o0 = env.get_ee_state()
        p0,o0 = env._pybullet.multiplyTransforms(p0, o0, [0.05, 0.065,-0.02], [0,0,0,1])
        angle = -0  # 90 degrees in radians
        rotation_quaternion = env._pybullet.getQuaternionFromEuler([0, 0, angle])
        new_pos, new_ori = env._pybullet.multiplyTransforms(p0, o0, [0,0,0], rotation_quaternion)
        base_orin = env._pybullet.getEulerFromQuaternion(new_ori)
        if base_link_id is None:
            base_link_shape = env._pybullet.createVisualShape(env._pybullet.GEOM_BOX, halfExtents=[0.025, 0.025, 0.03], rgbaColor=[0.2, 0.2, 0.2, 1])
            base_link_pos, base_link_ori = env._pybullet.multiplyTransforms(new_pos, new_ori, [0,-0.02,0], [0,0,0,1])
            base_link_id    = env._pybullet.createMultiBody(baseMass=0, baseCollisionShapeIndex=base_link_shape,
                                                        baseVisualShapeIndex=base_link_shape,
                                                        basePosition= base_link_pos , baseOrientation=base_link_ori)
        else:
            base_link_pos, base_link_ori = env._pybullet.multiplyTransforms(new_pos, new_ori, [0,-0.02,0.0], [0,0,0,1])
            env._pybullet.resetBasePositionAndOrientation(base_link_id, base_link_pos , base_link_ori)
        
        cam_pos = 0.8*cam_pos + 0.2*np.array([p0[0],0,0.2])
        
        env._pybullet.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=45, cameraPitch=-13, cameraTargetPosition=cam_pos)

        soft_robot_1.move_robot_ori(action=np.array([0.0, sf1_seg1_cable_1, sf1_seg1_cable_2, 
                                                    0.0, sf1_seg2_cable_1, sf1_seg2_cable_2,
                                                    sf1_seg3_cable_0, sf1_seg3_cable_1, sf1_seg3_cable_2]),
                                base_pos = new_pos, base_orin = base_orin)
        
