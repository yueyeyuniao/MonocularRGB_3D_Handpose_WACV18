# bala paen 1 , chap rast 2 , jelo aghab 3
# pylint: disable=wrong-import-position

import sys
sys.path.append("lib")
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import PyCeresIK as IK
from cv_bridge import CvBridge,CvBridgeError
from std_msgs.msg import String
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray


import time
import os
import numpy as np

from common import image
from common.opencv_grabbers import OpenCVGrabber
from common.calibrate import OpenCVCalib2CameraMeta, LoadOpenCVCalib
from common import mva19
from common import factory
from common import pipeline
import PyMBVCore  as Core
import PyJointTools as jt


from utils import detector_utils

"""
        limbSeq = [[0, 1], [0, 5], [0, 9], [0, 13], [0, 17], # palm
                [1, 2], [2, 3], [3,4], # thump
                [5, 6], [6, 7], [7, 8], # index
                [9, 10], [10, 11], [11, 12], # middle
                [13, 14], [14, 15], [15, 16], # ring
                [17, 18], [18, 19], [19, 20], # pinky
                ]
"""

class wrapper():

    def __init__(self, outSize, config, track=False, paused=False):
        
        self.config = config
        self.track=track
        self.paused = paused
        self.bridge = CvBridge()
        self.fake_clb = OpenCVCalib2CameraMeta(LoadOpenCVCalib("res/calib_webcam_mshd_vga.json"))

        print("Initialize WACV18 3D Pose estimator (IK)...")
        print(config)
        self.pose_estimator = factory.HandPoseEstimator(config)

        print("Initialize MVA19 CVRL Hand pose net...")
        self.estimator = mva19.Estimator(config["model_file"], config["input_layer"], config["output_layer"])
        self.detection_graph, self.sess = detector_utils.load_inference_graph()
        self.started = True
        self.delay = {True: 0, False: 1}
        self.p2d = self.bbox = None
        self.smoothing = self.config.get("smoothing", 0)
        self.boxsize = self.config["boxsize"]
        self.stride = self.config["stride"]
        self.peaks_thre = self.config["peaks_thre"]
        self.joints_publisher = rospy.Publisher('/hand_tracker/joints_pose' if len(sys.argv) < 3 else sys.argv[2], Float32MultiArray, queue_size=1 )
        
    def subscribe(self, topic):
        rospy.Subscriber(topic, Image, callback=self.callback)
    
    def _center(self, a, b):
        return [(v[0]+v[1])/2000 for v in list(zip(a,b)) ]


    def callback(self, img_msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            bgr = cv2.resize(bgr, (640,480), interpolation = cv2.INTER_AREA)
        except Exception as e:
            print("Failed to grab", e)
            return

        clb = self.fake_clb

        # compute kp using model initial pose
        points2d = self.pose_estimator.ba.decodeAndProject(self.pose_estimator.model.init_pose, clb)
        oldKp = np.array(points2d).reshape(-1, 2)


        score = -1
        result_pose = None
        # STEP2 detect 2D joints for the detected hand.
        if self.started:
            if self.bbox is None:
                self.bbox = detector_utils.hand_bbox(bgr,self.detection_graph,self.sess)
                if self.bbox is None:
                    cv2.imshow("3D Hand Model reprojection",bgr)
                    cv2.waitKey(1)
                    return
            else:
                dbox = detector_utils.hand_bbox(bgr,self.detection_graph,self.sess)
                if dbox is not None:
                    x,y,w,h = self.bbox
                    x1,y1,w1,h1 = dbox
                    if (x1>x+w or x1+w1<x ) or y1+h1<y or y1>y+h:
                        self.bbox = dbox
                        print("updated")
                    else:
                        x,y,w,h = dbox
                        b = max((w,h,224))
                        b = int(b + b*0.3)
                        cx = x + w/2
                        cy = y + h/2
                        x = cx-b/2
                        y = cy-b/2

                        x = max(0,int(x))
                        y = max(0,int(y))

                        x = min(x, bgr.shape[1]-b)
                        y = min(y, bgr.shape[0]-b)
                        
                        self.bbox = [x,y,b,b]

            x,y,w,h = self.bbox
            crop = bgr[y:y+h,x:x+w]
            img, pad = mva19.preprocess(crop, self.boxsize, self.stride)
            t = time.time()
            hm = self.estimator.predict(img)
            est_ms = (time.time() - t)

            # use joint tools to recover keypoints
            scale = float(self.boxsize) / float(crop.shape[0])
            scale = self.stride/scale
            ocparts = np.zeros_like(hm[...,0])
            peaks = jt.FindPeaks(hm[...,:-1], ocparts, self.peaks_thre, scale, scale)

            # convert peaks to hand keypoints
            hand = mva19.peaks_to_hand(peaks, x, y)

            if hand is not None:
                keypoints = hand
            
                mask = keypoints[:, 2] < self.peaks_thre
                keypoints[mask] = [0, 0, 1.0]

                if track:
                    keypoints[mask, :2] = oldKp[mask]

                keypoints[:, 2] = keypoints[:, 2]**3
                
                rgbKp = IK.Observations(IK.ObservationType.COLOR, clb, keypoints)
                obsVec = IK.ObservationsVector([rgbKp, ])
                score, res = self.pose_estimator.estimate(obsVec)
                
                if track:
                    result_pose = list(self.smoothing * np.array(self.pose_estimator.model.init_pose) + (1.0 - self.smoothing) * np.array(res))
                else:
                    result_pose = list(res)
                # score is the residual, the lower the better, 0 is best
                # -1 is failed optimization.
                if track:
                    if -1 < score: #< 150:
                        self.pose_estimator.model.init_pose = Core.ParamVector(result_pose)
                    else:
                        print("\n===>Reseting init position for IK<===\n")
                        self.pose_estimator.model.reset_pose()
                        self.bbox = None

                if score > -1:  # compute result points
                    self.p2d = np.array(self.pose_estimator.ba.decodeAndProject(Core.ParamVector(result_pose), clb)).reshape(-1, 2)
                    # scale = w/config.boxsize
                    self.bbox = mva19.update_bbox(self.p2d,bgr.shape[1::-1])
            
            p3d = np.array(self.pose_estimator.ba.decode(Core.ParamVector(result_pose), clb))
            joints_msg = Float32MultiArray()
            joints_msg.data = p3d.tolist()
            self.joints_publisher.publish(joints_msg)

        viz = np.copy(bgr)
        if self.started and result_pose is not None:
            viz = mva19.visualize_3dhand_skeleton(viz, self.p2d)
            pipeline.draw_rect(viz, "Hand", self.bbox, box_color=(0, 255, 0), text_color=(200, 200, 0))
            cv2.putText(viz, 'Hand pose estimation', (20, 20), 0, 0.7, (50, 20, 20), 1, cv2.LINE_AA)
            cv2.imshow("3D Hand Model reprojection", viz)

        key = cv2.waitKey(self.delay[self.paused])
        if key & 255 == ord('p'):
            self.paused = not self.paused
        if key & 255 == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)
        if key & 255 == ord('r'):
            print("\n===>Reseting init position for IK<===\n")
            self.pose_estimator.model.reset_pose()
            self.bbox = None
            print("RESETING BBOX",self.bbox)





if __name__ == "__main__":
    # global config,outSize,track,paused, with_renderer,pub,transform
    
    rospy.init_node('hand-tracker')
    track=True
    paused=False
    outSize = (640,480)
    config = {
        "model": "models/hand_skinned.xml", "model_left": False,
        "model_init_pose": [-109.80840809323652, 95.70022984677065, 584.613931114289, 292.3322807284121, -1547.742897973965, -61.60146881490577, 435.33025195547793, 1.5707458637241434, 0.21444030289465843, 0.11033385117688158, 0.021952050059337137, 0.5716581133215294, 0.02969734913698679, 0.03414155945643072, 0.0, 1.1504613679382742, -0.5235922979328, 0.15626331136368257, 0.03656410417088128, 8.59579088582312e-07, 0.35789633949684985, 0.00012514308785717494, 0.005923001258945023, 0.24864102398139007, 0.2518954858979162, 0.0, 3.814694400000002e-13],
        "model_map": IK.ModelAwareBundleAdjuster.HAND_SKINNED_TO_OP_RIGHT_HAND,

        "ba_iter": 100,
        "padding": 0.3,
        "minDim": 170,

        "smoothing": 0.2,

        "model_file": "models/mobnet4f_cmu_adadelta_t1_model.pb",
        "input_layer": "input_1",
        "output_layer": "k2tfout_0",
        "stride": 4,
        "boxsize": 224,
        "peaks_thre": 0.1,
        
        # default bbox for the hand location
        # "default_bbox": [322, 368, 110, 109],
    }
    
    wrapper = wrapper((640,480), config, track=track)
    wrapper.subscribe('/image_raw' if len(sys.argv) < 2 else sys.argv[1])
    rospy.spin()
    cv2.destroyAllWindows()

