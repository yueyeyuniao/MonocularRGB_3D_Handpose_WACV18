'''
Adapted from the MonoHand3D codebase for the MonocularRGB_3D_Handpose project (github release)

This script uses the 2D joint estimator of Gouidis et al. 

@author: Paschalis Panteleris (padeler@ics.forth.gr)
'''

import sys
sys.path.append("lib")
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3

import time
import os

import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import numpy as np

import PyCeresIK as IK

from common import image
from common.opencv_grabbers import OpenCVGrabber
from common.calibrate import OpenCVCalib2CameraMeta, LoadOpenCVCalib

from common import factory
from common import pipeline


import PyMBVCore  as Core
import PyJointTools as jt

from common import mva19 


def mono_hand_loop(acq, outSize, config, track=False, paused=False, with_renderer=False):

    print("Initialize WACV18 3D Pose estimator (IK)...")
    pose_estimator = factory.HandPoseEstimator(config)

    if with_renderer:
        print("Initialize Hand Visualizer...")
        hand_visualizer = pipeline.HandVisualizer(factory.mmanager, outSize)

    print("Initialize MVA19 CVRL Hand pose net...")
    estimator = mva19.Estimator(config["model_file"], config["input_layer"], config["output_layer"])

    left_hand_model = config["model_left"]
    started = False
    delay = {True: 0, False: 1}
    ik_ms = est_ms = 0
    p2d = bbox = None
    count = 0
    mbv_viz = opviz = None
    smoothing = config.get("smoothing", 0)
    boxsize = config["boxsize"]
    stride = config["stride"]
    peaks_thre = config["peaks_thre"]
    print("Entering main Loop.")

    while True:
        try:
            imgs, clbs = acq.grab()
            if imgs is None or len(imgs)==0:
                break
        except Exception as e:
            print("Failed to grab", e)
            break

        st = time.time()
        bgr = imgs[0]
        clb = clbs[0]

        # compute kp using model initial pose
        points2d = pose_estimator.ba.decodeAndProject(pose_estimator.model.init_pose, clb)
        oldKp = np.array(points2d).reshape(-1, 2)

        if bbox is None:
            bbox = config["default_bbox"]

        score = -1
        result_pose = None
        crop_viz = None

        # STEP2 detect 2D joints for the detected hand.
        if started and bbox is not None:
            x,y,w,h = bbox
            # print("BBOX: ",bbox)
            crop = bgr[y:y+h,x:x+w]
            img, pad = mva19.preprocess(crop, boxsize, stride)

            t = time.time()
            hm = estimator.predict(img)
            est_ms = (time.time() - t)
        
            # use joint tools to recover keypoints
            scale = float(boxsize) / float(crop.shape[0])
            scale = stride/scale
            ocparts = np.zeros_like(hm[...,0])
            peaks = jt.FindPeaks(hm[...,:-1], ocparts, peaks_thre, scale, scale)

            # convert peaks to hand keypoints
            hand = mva19.peaks_to_hand(peaks, x, y)

            if hand is not None:
                keypoints = hand
            
                mask = keypoints[:, 2] < peaks_thre
                keypoints[mask] = [0, 0, 1.0]

                if track:
                    keypoints[mask, :2] = oldKp[mask]

                keypoints[:, 2] = keypoints[:, 2]**3
                
                rgbKp = IK.Observations(IK.ObservationType.COLOR, clb, keypoints)
                obsVec = IK.ObservationsVector([rgbKp, ])
                t = time.time()
                score, res = pose_estimator.estimate(obsVec)
                ik_ms = (time.time() - t)
                # print(count,)
                pose_estimator.print_report()

                if track:
                    result_pose = list(smoothing * np.array(pose_estimator.model.init_pose) + (1.0 - smoothing) * np.array(res))
                else:
                    result_pose = list(res)

                # score is the residual, the lower the better, 0 is best
                # -1 is failed optimization.
                if track:
                    if -1 < score:# < 20000:
                        pose_estimator.model.init_pose = Core.ParamVector(result_pose)
                    else:
                        print("\n===>Reseting init position for IK<===\n")
                        pose_estimator.model.reset_pose()

                if score > -1:  # compute result points
                    p2d = np.array(pose_estimator.ba.decodeAndProject(Core.ParamVector(result_pose), clb)).reshape(-1, 2)
                    # scale = w/config.boxsize
                    bbox = mva19.update_bbox(p2d,bgr.shape[1::-1])



        viz = np.copy(bgr)
        viz2d = np.zeros_like(bgr)
        if started and result_pose is not None:
            viz2d = mva19.visualize_2dhand_skeleton(viz2d, hand, thre=peaks_thre)
            cv2.imshow("2D CNN estimation",viz2d)
            header = "FPS OPT+VIZ %03d, OPT %03d (CNN %03d, 3D %03d)"%(1/(time.time()-st),1/(est_ms+ik_ms),1.0/est_ms, 1.0/ik_ms) 
            
            if with_renderer:
                hand_visualizer.render(pose_estimator.model, Core.ParamVector(result_pose), clb)
                mbv_viz = hand_visualizer.getDepth()
                cv2.imshow("MBV VIZ", mbv_viz)
                mask = mbv_viz != [0, 0, 0]
                viz[mask] = mbv_viz[mask]
            else:
                viz = mva19.visualize_3dhand_skeleton(viz, p2d)
                pipeline.draw_rect(viz, "Hand", bbox, box_color=(0, 255, 0), text_color=(200, 200, 0))


        else:
            header = "Press 's' to start, 'r' to reset pose, 'p' to pause frame."
        


        cv2.putText(viz, header, (20, 20), 0, 0.7, (50, 20, 20), 1, cv2.LINE_AA)
        cv2.imshow("3D Hand Model reprojection", viz)

        key = cv2.waitKey(delay[paused])
        if key & 255 == ord('p'):
            paused = not paused
        if key & 255 == ord('q'):
            break
        if key & 255 == ord('r'):
            print("\n===>Reseting init position for IK<===\n")
            pose_estimator.model.reset_pose()
            bbox = config['default_bbox']
            print("RESETING BBOX",bbox)
        if key & 255 == ord('s'):
            started = not started


        count += 1




if __name__ == '__main__':

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
        "default_bbox": [170,80,300,300],
    }
    
    # NOTE: You can replace the camera id with a video filename. 
    acq = OpenCVGrabber("web1.mp4", calib_file="res/calib_webcam_mshd_vga.json")
    acq.initialize()
    mono_hand_loop(acq, (640,480), config,  track=True, with_renderer=True)

