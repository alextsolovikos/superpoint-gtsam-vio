# import VisualInertialOdometry as vio
import pykitti
import matplotlib.pyplot as plt
import argparse
import superpoint as sp
import numpy as np
import cv2

if __name__ == '__main__':
    # For testing, use the following command in superpoint-gtsam-vio/src:
    #    python3 main.py --basedir data --date '2011_09_26' --drive '0005'
    parser = argparse.ArgumentParser(description='Visual Inertial Odometry of KITTI dataset.')
    parser.add_argument('--basedir', dest='basedir', type=str)
    parser.add_argument('--date', dest='date', type=str)
    parser.add_argument('--drive', dest='drive', type=str)
    parser.add_argument('--max_length', dest='max_length', type=int)
    args = parser.parse_args()

    """ 
    Use pykitti to get raw data
    """
    raw_data = pykitti.raw(args.basedir, args.date, args.drive)


    """
    Run superpoint to get keypoints
    """
    print('==> Loading pre-trained network.')
    # This class runs the SuperPoint network and processes its outputs.
    # Inputs from list of default options in superpoint_demo.py.
    fe = sp.SuperPointFrontend(weights_path='superpoint_v1.pth',
                            nms_dist=4,
                            conf_thresh=0.015,
                            nn_thresh=0.7,
                            cuda=False)
    print('==> Successfully loaded pre-trained network.')

    # This class helps merge consecutive point matches into tracks.
    tracker = sp.PointTracker(max_length=args.max_length, nn_thresh=fe.nn_thresh)

    print('==> Running SuperPoint')
    skip = 5; # Images recorded roughly every 0.1 s (see timestamps file)
    N = len(raw_data.timestamps)
    idx = range(0, N, skip);
    for i in idx:
        img, _ = raw_data.get_gray(i) # only get image from cam0
        img_np = np.array(img).astype('float32') / 255.0;
        pts, desc, _ = fe.run(img_np)
        tracker.update(pts, desc)

    vision_data = tracker.get_vision_data();


    """
    Setup GTSAM factor graph with imu measurements and keypoints
    """

    """
    Solve factor graph
    """

    """
    Visualize results
    """













