#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np  # add numpy import

class ImageSubscriber(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/rgb1", Image, self.callback)
        self.save = True

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv2.imshow("Subscribed Image", cv_image)
        if self.save:
            image_shape = np.shape(cv_image)  # get shape of the image
            print("Image size: ", image_shape)  # print the shape
            print("image is saved")
            cv2.imwrite('/home/kblee/catkin_ws/src/flightmare/flightros/src/practice/target_drone.jpg', cv_image)
            self.save = False
        cv2.waitKey(3)

def main():
    rospy.init_node('image_subscriber', anonymous=True)
    image_subscriber = ImageSubscriber()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()