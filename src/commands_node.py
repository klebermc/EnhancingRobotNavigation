#!/usr/bin/env python3
import rospy
import numpy as np
import onnxruntime as ort
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32MultiArray
from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal
from core import ExclusionZone
from Auxiliar import *
import os

experiment = {
    "dt": 0.1,#
    "env_size": 100,
    "win_radius": 5,
    "evaluation_episodes" : 10,
    "type": "dynamic",
    "subtype": "equal",
    "H": 10, #prediction horizon
    "weightsEqual": [1/7,1/7,1/7,1/7,1/7,1/7,1/7],
    "weightsA": [0.2,0.2,0.2,0.1 ,0.1 ,0.1 ,0.1 ],
    "weightsB": [0.1,0.1,0.1,0.25,0.25,0.1 ,0.1 ],
    "weightsC": [0.1,0.1,0.1,0.1 ,0.1 ,0.25,0.25],
}

commands = Twist()
commandsS = TwistStamped()
internal_values = TwistStamped()

class CommandsNode():
    def generate_observation(self, OBSTs, pursuer, evader):

        stateE = evader

        maxD=2*5*np.sqrt(2)

        observation=[]
        stateP = pursuer

        # ------- evader to pursuer ----------
        pos_pur_eva = stateE[0:2] - stateP[0:2]
        dist_pur_eva = np.linalg.norm(pos_pur_eva)
        angle = np.arctan2(pos_pur_eva[1], pos_pur_eva[0])
        dist_pur_eva /= maxD 
        angle /= np.pi
        pTheta= stateP[2]/np.pi


        # ------- pursuers to ezs ----------
        dPEzs=[]
        angPEzs=[]

        for ez in OBSTs:
            vecPEzi = ez.pos[0:2]-stateP[0:2]
            dwez = np.clip((np.linalg.norm(vecPEzi)-ez.radius) / maxD, -1, 1)
            dPEzs.append(dwez) 
            angPEzs.append(np.arctan2(vecPEzi[1], vecPEzi[0])/np.pi) 


        dist_pur_eva = np.clip(dist_pur_eva, 0, 1)
        observation.append([dist_pur_eva, angle, pTheta, dPEzs, angPEzs])

        flattened_observation = flatten(observation)
        observation_array = np.array(flattened_observation)
        
        return observation_array

    def trajectory_metrics(self, traj, evader, OBSTs):
        H=len(traj)
        metrics ={}
        # A.1 Obstacle safety  the percentage of time spent by the robot in the dangerous area around obstacle
        threshold=1
        time_in_dangerous_area = 0
        for i in range(H):
            for i,obst in enumerate(OBSTs):
                if np.linalg.norm(traj[i][:2] - obst.pos[:2]) < obst.radius + threshold and i<len(OBSTs)-1: # I don't consider the last obstacle as an object, but as a person
                    # print("Dangerous area")
                    time_in_dangerous_area += 1
                    break # do not check other obsts
        metrics["OS"] = time_in_dangerous_area/H

        # A.2 Motion efficiency - estimated time to reach the target

        metrics["METD"] = np.sum(np.linalg.norm(np.diff(np.array(traj)[:,:2], axis=0), axis=1))/(2*5*np.sqrt(2))
        metrics["ME"] = (np.linalg.norm(traj[-1][:2]-evader[:2])  + np.sum(np.linalg.norm(np.diff(np.array(traj)[:,:2], axis=0), axis=1)) )/(2*5*np.sqrt(2))

        # A.4 Cumulative heading change - the sum of the heading changes over the control horizon
        metrics["CHC"] = np.sum(np.abs(np.diff(np.array([a[2] for a in traj]))))

        # B.1 Velocity smoothness - how much robot linear velocities vx and vy changed over the control horizon 
        t = np.array(traj)[:, :2]
        metrics["VS"] = np.sum(np.sqrt(np.sum(np.diff(t, axis=0)**2, axis=1))/1)/(H-1)
                            
        # B.2 Heading change smoothness - The m hsm metric is computed by comparing differences of robot angular velocity
        metrics["HCS"] = np.sum(np.abs(np.diff(np.array([a[2] for a in traj]))))/(H-1)

        # C.1 Personal spaces intrusion - defines the scale of robot intrusions into any humans personal space
        #humans heading hthetan
        hvarnfr = 0.5#0.01 #v=0.5
        hvarnrr = (1/2)*hvarnfr
        hvarnsd = (2/3)*hvarnfr

        #Variances along the front ( h var n fr ), side ( h var n sd ), and rear ( h var n rr ) directions of the human pose
        #r,h var n hd , is selected (h var n fr or h var n rr ) in a three-step procedure:
        
        den=0
        hthetan = 0
        for n in range (H):
            rxn = traj[n][0]/5
            ryn = traj[n][1]/5

            maxrhpsin=[]
            # for obst in OBSTs: # I don't consider the last obstacle as an object, but as a person
            obst = OBSTs[-1]
            hxn=obst.pos[0]/5
            hyn=obst.pos[1]/5

            # eq 8 - 10
            rhphin = np.arctan2(ryn - hyn, rxn - hxn)
            rhdn = rhphin - hthetan
            if np.abs(rhdn) <= np.pi/2:
                rhvarn_hd = hvarnfr
            else:
                rhvarn_hd = hvarnrr

            # eqs 11 - 12
            rhsumnpsi = np.matmul(np.identity(2),np.matmul(np.array([[rhvarn_hd,0],[0,hvarnsd]]),np.identity(2)))
            # hsumnp - The covariance matrix of the estimated human position obtained from the robot perception system.
            hsumnp = np.identity(2)*0.0
            rhdsumnpsi = hsumnp + rhsumnpsi

            #the symmetrical variant of the multivariate Gaussian, f mg
            # inputs of fmg are a pose and a multivariate normal distribution, the value of which will be computed at the given pose
            rhpsin = multivariate_normal.pdf([rxn,ryn], mean=[hxn,hyn], cov=rhdsumnpsi)
            hpsin = multivariate_normal.pdf([hxn,hyn], mean=[hxn,hyn], cov=rhdsumnpsi)
            maxrhpsin.append(rhpsin/hpsin)
            # r,h psi n - scale of r robot intrusion into the personal space of h-th human in time tn
            # robots pose at time tn, rpn
            den += np.max(np.array(maxrhpsin))
        metrics["PSI"] = den /H
        
        # C.3 Heading straight into a human (or obstacles)
        robot=traj[-1]
        hsih=[]
        for obst in OBSTs:
            vector1 = obst.pos[:2]-robot[:2]
            vector2 = np.array([np.cos(robot[2]), np.sin(robot[2])])
            unit_vector1 = vector1 / np.linalg.norm(vector1)
            unit_vector2 = vector2 / np.linalg.norm(vector2)
            dot_product = np.dot(unit_vector1, unit_vector2)
            angle = np.arccos(dot_product) #angle in radian
            hsih.append( (1 - abs(angle)/(np.pi/2))*0)

        metrics["HSIH"] = np.max(np.array(hsih))
        
        rospy.loginfo('OS %f ME %f CHC %f VS %f HCS %f PSI %f HSIH %f',
                      metrics["OS"],metrics["ME"],
                      metrics["CHC"],metrics["VS"],
                      metrics["HCS"],metrics["PSI"],
                      metrics["HSIH"])

        # ic(metrics)

        return metrics

    def robot_model(self, control, state, dt, speed):
        control*=np.pi
        robot = state + dt*np.array([speed*np.cos(state[2]), speed*np.sin(state[2]), control])
        return robot

    def __init__(self):
        #self.drone_pose = PoseStamped()
        #self.drone_vel = TwistStamped()
        self.obs = np.zeros((1,9))
        self.target = np.array([2,0])
        self.obstacle1 = np.array([-1,-1.5]) #np.zeros((2))
        self.obstacle2 = np.array([1,-1.5]) #np.zeros((2))
        self.obstacle3 = np.array([0,0.5]) #np.zeros((2))
        self.vel = np.array([1,0])
        self.heading = np.arctan2(self.vel[1],self.vel[0])/np.pi
        

        self.marker1 = Marker()
        self.marker1.header.frame_id = "world"
        self.marker1.header.stamp = rospy.Time.now()
        self.marker1.ns = "spheres"
        self.marker1.id = 0
        self.marker1.type = Marker.SPHERE
        self.marker1.action = Marker.ADD
        self.marker1.pose.position.x = self.obstacle1[0]
        self.marker1.pose.position.y = self.obstacle1[1]
        self.marker1.pose.position.z = 0.0
        self.marker1.pose.orientation.x = 0.0
        self.marker1.pose.orientation.y = 0.0
        self.marker1.pose.orientation.z = 0.0
        self.marker1.pose.orientation.w = 1.0
        self.marker1.scale.x = 0.5
        self.marker1.scale.y = 0.5
        self.marker1.scale.z = 0.5
        self.marker1.color.a = 1.0# Alpha (opacity)
        self.marker1.color.r = 0.45# Red
        self.marker1.color.g = 0.45# Green
        self.marker1.color.b = 0.45# Blue# Define the second sphere
        self.marker2 = Marker()
        self.marker2.header.frame_id = "world"
        self.marker2.header.stamp = rospy.Time.now()
        self.marker2.ns = "spheres"
        self.marker2.id = 1
        self.marker2.type = Marker.SPHERE
        self.marker2.action = Marker.ADD
        self.marker2.pose.position.x = self.obstacle2[0]
        self.marker2.pose.position.y = self.obstacle2[1]
        self.marker2.pose.position.z = 0.0
        self.marker2.pose.orientation.x = 0.0
        self.marker2.pose.orientation.y = 0.0
        self.marker2.pose.orientation.z = 0.0
        self.marker2.pose.orientation.w = 1.0
        self.marker2.scale.x = 0.5
        self.marker2.scale.y = 0.5
        self.marker2.scale.z = 0.5
        self.marker2.color.a = 1.0# Alpha (opacity)
        self.marker2.color.r = 0.45# Red
        self.marker2.color.g = 0.45# Green
        self.marker2.color.b = 0.45# Blue# Publish the markers
        self.marker3 = Marker()
        self.marker3.header.frame_id = "world"
        self.marker3.header.stamp = rospy.Time.now()
        self.marker3.ns = "spheres"
        self.marker3.id = 2
        self.marker3.type = Marker.SPHERE
        self.marker3.action = Marker.ADD
        self.marker3.pose.position.x = self.obstacle3[0]
        self.marker3.pose.position.y = self.obstacle3[1]
        self.marker3.pose.position.z = 0.0
        self.marker3.pose.orientation.x = 0.0
        self.marker3.pose.orientation.y = 0.0
        self.marker3.pose.orientation.z = 0.0
        self.marker3.pose.orientation.w = 1.0
        self.marker3.scale.x = 0.5
        self.marker3.scale.y = 0.5
        self.marker3.scale.z = 0.5
        self.marker3.color.a = 0.8# Alpha (opacity)
        self.marker3.color.r = 0.0# Red
        self.marker3.color.g = 1.0# Green
        self.marker3.color.b = 0.0# Blue# Publish the markers
        self.marker4 = Marker()
        self.marker4.header.frame_id = "world"
        self.marker4.header.stamp = rospy.Time.now()
        self.marker4.ns = "spheres"
        self.marker4.id = 3
        self.marker4.type = Marker.SPHERE
        self.marker4.action = Marker.ADD
        self.marker4.pose.position.x = 2.0
        self.marker4.pose.position.y = 0.0
        self.marker4.pose.position.z = 0.0
        self.marker4.pose.orientation.x = 0.0
        self.marker4.pose.orientation.y = 0.0
        self.marker4.pose.orientation.z = 0.0
        self.marker4.pose.orientation.w = 1.0
        self.marker4.scale.x = 0.25
        self.marker4.scale.y = 0.25
        self.marker4.scale.z = 0.25
        self.marker4.color.a = 1.0# Alpha (opacity)
        self.marker4.color.r = 0.0# Red
        self.marker4.color.g = 0.0# Green
        self.marker4.color.b = 1.0# Blue# Publish the markers

        self.drone_pose_sub = rospy.Subscriber("/vrpn_client_node/Spot/pose",PoseStamped,self.robot_pose_sub_cb)
        # self.obstacle1_sub = rospy.Subscriber("/vrpn_client_node/Obstacle1/pose",PoseStamped,self.obstacle1_sub_cb)
        # self.obstacle2_sub = rospy.Subscriber("/vrpn_client_node/Obstacle2/pose",PoseStamped,self.obstacle2_sub_cb)
        # self.obstacle3_sub = rospy.Subscriber("/tarot1/mavros/vision_pose/pose",PoseStamped,self.obstacle3_sub_cb)
        self.commands_pub = rospy.Publisher('/spot/slow/cmd_vel',Twist,queue_size=1)
        self.commands_pubS = rospy.Publisher('/spot/slow/cmd_vel_stamped',TwistStamped,queue_size=1)
        self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        self.internal_values_pub = rospy.Publisher('/internal_values',TwistStamped,queue_size=1)

        
        self.OBSTs = []
        radius = 0.5
        for i in range(3):
            ez_name = "OBST_"+str(i)
            self.OBSTs.append(ExclusionZone(radius,5,ez_name))

        self.OBSTs[0].pos[0]=self.obstacle1[0]
        self.OBSTs[0].pos[1]=self.obstacle1[1]
        self.OBSTs[1].pos[0]=self.obstacle2[0]
        self.OBSTs[1].pos[1]=self.obstacle2[1]
        self.OBSTs[2].pos[0]=self.obstacle3[0]
        self.OBSTs[2].pos[1]=self.obstacle3[1]

        dir_models = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.tactics = []
        self.tactics.append(tactic([],os.path.join(dir_models, "tactic1", "tactic1"),1))
        self.tactics.append(tactic([],os.path.join(dir_models, "tactic2", "tactic2"),1))
        self.tactics.append(tactic([],os.path.join(dir_models, "tactic3", "tactic3"),1))

        #start a timer counter
        self.selected_tactic = np.zeros(100000)-1
        self.drone_pose=np.zeros(3)
        self.t=0


    def obstacle1_sub_cb(self,pose):
        #self.drone_pose = pose
        self.obstacle1 = np.array([pose.pose.position.x,pose.pose.position.y])

    def obstacle2_sub_cb(self,pose):
        #self.drone_pose = pose
        self.obstacle2 = np.array([pose.pose.position.x,pose.pose.position.y])

    def obstacle3_sub_cb(self,pose):
        #self.drone_pose = pose
        self.obstacle3 = np.array([pose.pose.position.x,pose.pose.position.y])
        
    def robot_pose_sub_cb(self,pose):
        #self.drone_pose = pose
        drone_position = np.array([pose.pose.position.x,pose.pose.position.y])
        target = self.target - drone_position
        target_pol = self.cart2pol(target)
        obst1 = self.obstacle1 - drone_position
        obst1_pol = self.cart2pol(obst1)
        obst2 = self.obstacle2 - drone_position
        obst2_pol = self.cart2pol(obst2)
        obst3 = self.obstacle3 - drone_position
        obst3_pol = self.cart2pol(obst3)
        heading_q = R.from_quat([pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w])
        #self.heading = heading_q.as_euler('zyx', degrees=True)[0]
        inflation = 0.8
        
        # self.obs = np.array([[target_pol[0], target_pol[1], self.heading, inflation*obst1_pol[0], inflation*obst2_pol[0],inflation*obst3_pol[0]
                            #   , obst1_pol[1], obst2_pol[1], obst3_pol[1]]])
        
        # rospy.loginfo( pose.pose.position.x)
        # rospy.loginfo( pose.pose.position.y)
        # rospy.loginfo( self.heading)
        self.drone_pose = np.array([pose.pose.position.x,pose.pose.position.y,self.heading])
        self.obs = self.generate_observation(self.OBSTs, self.drone_pose, self.target)

        # rospy.loginfo('d, %f, pol, %f, hd %f', target_pol[0], target_pol[1], self.heading)
        # rospy.loginfo('obs 1, %f, obs 2, %f, obs 3 %f', obst1_pol[1]*180, obst2_pol[1]*180, obst3_pol[1]*180)

    def cart2pol(self,cart):
        theta = np.arctan2(cart[1],cart[0])/np.pi #normalizing angles
        r = np.linalg.norm(cart)/(np.sqrt(2*5**2)) #dividing by the maximum diagonal of the test area to normalize
        return np.array([r,theta])
    
    def inference(self):
        # ort_sess = ort.InferenceSession('my_sac_actor.onnx')
        # outputs = ort_sess.run(None, {'input':self.obs.astype(np.float32)})
        # return outputs[0][0][0]

        # if the timer is greater than a threshold, select a new tactic
        if self.t%2 == 0:
            
            cost_k=[]
            metrics_k=[]
            # self.selected_tactic = np.random.randint(0,3)
            for tau in self.tactics:
                robot=self.drone_pose #initial states
                sim_trajectory=[]
                #simulating the robot over a control horizon
                simObs=self.obs.copy()
                for h in range(1,experiment["H"]):
                    #computing the action
                    action = tau.compute_Action(simObs)
                    # ic(action)

                    #evolve state
                    robot =  self.robot_model(action, robot,0.2, 0.5)
                    # observe
                    simObs = self.generate_observation(self.OBSTs, self.drone_pose, self.target)
                    
                    #store the trajectory
                    sim_trajectory.append(robot.copy())
                    
                # ic(sim_trajectory)
                #now that we must compute the metrics on that trajectory
                metrics = self.trajectory_metrics(sim_trajectory, self.target, self.OBSTs)
                
                #objective function with weights for metrics
                if experiment["subtype"].lower() == "equal":
                    w=experiment["weightsEqual"]
                elif experiment["subtype"] == "A":
                    w=experiment["weightsA"]
                elif experiment["subtype"] == "B":
                    w=experiment["weightsB"]
                elif experiment["subtype"] == "C":
                    w=experiment["weightsC"]
                
                cost_k.append( w[0] * metrics["OS"] + w[1] * metrics["ME"] + w[2] * metrics["OS"] + w[3] * metrics["VS"] + w[4] * metrics["HCS"] + w[5] * metrics["PSI"] + w[6] * metrics["HSIH"])
                metrics_k.append(metrics.copy())

            self.selected_tactic[self.t] = np.argmin(cost_k)

        action = np.array(self.tactics[int(self.selected_tactic[self.t])].compute_Action(self.obs) , ndmin=2).squeeze()
        self.selected_tactic[self.t+1] = self.selected_tactic[self.t]
        rospy.loginfo('t %f, sel t %f',self.t, self.selected_tactic[self.t])

        self.t+=1

        return action
    
    def calc_vel(self):
        vel=0.25

        angular_velocity = self.inference()
        u = np.arctan2(self.vel[1],self.vel[0]) + 0.1*angular_velocity*np.pi
        u = np.arctan2(np.sin(u),np.cos(u))
        vx = vel*np.cos(u)
        vy = vel*np.sin(u)
        commands.linear.x = vx
        commands.linear.y = vy
        self.commands_pub.publish(commands)

        commandsS.twist.linear.x = vx
        commandsS.twist.linear.y = vy
        commandsS.header.stamp = rospy.Time.now()
        commandsS.header.frame_id = "Spot"
        self.commands_pubS.publish(commandsS)
        self.vel = np.array([vx,vy])  

        # angular_velocity = 1*self.inference()*np.pi
        # commands.linear.x = vel
        # commands.angular.z = angular_velocity
        # self.commands_pub.publish(commands)

        # commandsS.twist.linear.x = vel
        # commandsS.twist.angular.z = angular_velocity
        # commandsS.header.stamp = rospy.Time.now()
        # commandsS.header.frame_id = "Spot"
        # self.commands_pubS.publish(commandsS)

        # rospy.loginfo('vel x %f, vel y %f, angle %f, ang vel %f',vx,vy, u*180/np.pi, angular_velocity)
        self.heading = np.arctan2(self.vel[1],self.vel[0])/np.pi 
        self.marker_pub.publish(self.marker1)
        self.marker_pub.publish(self.marker2)
        self.marker_pub.publish(self.marker3)
        self.marker_pub.publish(self.marker4)

        
        internal_values.twist.linear.x = angular_velocity
        internal_values.twist.linear.y = self.selected_tactic[self.t]
        internal_values.twist.linear.z = 0
        internal_values.twist.angular.x = 0
        internal_values.twist.angular.y = 0
        internal_values.twist.angular.z = 0
        internal_values.header.stamp = rospy.Time.now()
        internal_values.header.frame_id = "Spot"
        self.internal_values_pub.publish(internal_values)

if __name__ == '__main__':
    rospy.init_node('commands_node', anonymous=True)
    
    commander = CommandsNode()
    rate = rospy.Rate(5)
    vel = np.array([1,0])
    while not rospy.is_shutdown():
        commander.calc_vel()
        rate.sleep()
