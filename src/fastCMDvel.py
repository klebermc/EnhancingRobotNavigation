#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class FastCMDVel():
    def __init__(self):
        self.sub = rospy.Subscriber("/spot/slow/cmd_vel", Twist, self.callbackSlowMessage)
        self.pub = rospy.Publisher('/spot/cmd_vel', Twist, queue_size=10)
        self.rate = rospy.Rate(20) 
        self.cmd = Twist()
    
    def callbackSlowMessage(self, data):
        self.cmd = data

    def talker(self):
        hello_str = "Started Publisher %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        while not rospy.is_shutdown():
            self.pub.publish(self.cmd)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        rospy.init_node('FastCMDVel', anonymous=True)
        fastcmdvel = FastCMDVel()
        fastcmdvel.talker()
    except rospy.ROSInterruptException:
        quit()