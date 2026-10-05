#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from visualization_msgs.msg import Marker
import numpy as np
import pickle



class Listener():
    def __init__(self):
        # Global list to store position data
        self.position_data = []
        self.velocity_data = []
        self.marker_data = {0:[], 1:[], 2:[]}
        
        rospy.Subscriber('/vrpn_client_node/Spot/pose', PoseStamped, self.callbackPOS)
        rospy.Subscriber('/spot/slow/cmd_vel', Twist, self.callbackVEL)
        rospy.Subscriber('/visualization_marker', Marker, self.callbackMarker)
        
    def callbackMarker(self,msg):
        """
        Callback function to extract position data from PoseStamped message.
        """
        # Extract position (x, y, z) from the PoseStamped message
        marker = msg
        if marker.id<3:
            marker_array = [marker.pose.position.x, marker.pose.position.y]

            # Append the position data to the global list
            self.marker_data[marker.id].append(marker_array)
            rospy.loginfo(f"Marker: {marker_array}")
        
    def callbackPOS(self,msg):
        """
        Callback function to extract position data from PoseStamped message.
        """
        # Extract position (x, y, z) from the PoseStamped message
        position = msg.pose.position
        position_array = [position.x, position.y, np.arctan2(self.velocity_data[-1][1],self.velocity_data[-1][0])]

        # Append the position data to the global list
        self.position_data.append(position_array)
        rospy.loginfo(f"Received position: {position_array}")


    def callbackVEL(self,msg):
        """
        Callback function to extract velocity data from Twist message.
        """
        # Extract velocity (vx, vy) from the Twist message
        velocity = msg.linear
        velocity_array = [velocity.x, velocity.y]

        # Append the position data to the global list
        self.velocity_data.append(velocity_array)
        rospy.loginfo(f"Received velocity: {velocity_array}")

    def save_position_data(self):
        """
        Save the data to a file using pickle.
        """
        data=(np.array(self.position_data),np.array(self.velocity_data),self.marker_data)
        with open('experiment_data.pkl', 'wb') as f:
            pickle.dump( data, f)
        rospy.loginfo("Data saved to position_data.pkl")

    def listener(self):
       

        # Set the loop rate (e.g., 10 Hz)
        rate = rospy.Rate(10)

        # Continue to listen for messages until the node is shutdown
        while not rospy.is_shutdown():
            # Process incoming messages
            rate.sleep()

        # Save position data when shutting down
        self.save_position_data()

if __name__ == '__main__':
    try:
        """
        ROS listener to subscribe to the PoseStamped topic.
        """
        # Initialize the ROS node
        rospy.init_node('Topic2Pickle', anonymous=True)
    
        l = Listener()
        l.listener()
    except rospy.ROSInterruptException:
        pass
