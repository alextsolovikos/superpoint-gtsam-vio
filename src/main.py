# import VisualInertialOdometry as vio
import pykitti
import matplotlib.pyplot as plt
import argparse
import SuperPointPretrainedNetwork.demo_superpoint as sp
import numpy as np
import cv2

def get_vision_data(tracker):
    """ Get keypoint-data pairs from the tracks. 
    """
    # Store the number of points per camera.
    pts_mem = tracker.all_pts
    N = len(pts_mem) # Number of cameras/images.
    # Get offset ids needed to reference into pts_mem.
    offsets = tracker.get_offsets()
    # Iterate through each track and get the data from the current image.
    vision_data = -1 * np.ones((tracker.tracks.shape[0], N, 2), dtype=int)
    print('Size of vision_data: ', vision_data.shape)
    for j, track in enumerate(tracker.tracks):
      for i in range(N-1):
        if track[i+3] == -1: # track[i+2] == -1 or 
          continue
        offset2 = offsets[i+1]
        idx2 = int(track[i+3]-offset2)
        pt2 = pts_mem[i+1][:2, idx2]
        vision_data[j, i] = np.array([int(round(pt2[0])), int(round(pt2[1]))])
    return vision_data


if __name__ == '__main__':
    # For testing, use the following command in superpoint-gtsam-vio/src:
    #    python3 main.py --basedir data --date '2011_09_26' --drive '0005'
    parser = argparse.ArgumentParser(description='Visual Inertial Odometry of KITTI dataset.')
    parser.add_argument('--basedir', dest='basedir', type=str)
    parser.add_argument('--date', dest='date', type=str)
    parser.add_argument('--drive', dest='drive', type=str)
    parser.add_argument('--max_length', dest='max_length', type=int, default=5)
    parser.add_argument('--n_skip', dest='n_skip', type=int, default=1)
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
    fe = sp.SuperPointFrontend(weights_path='src/SuperPointPretrainedNetwork/superpoint_v1.pth',
                            nms_dist=4,
                            conf_thresh=0.015,
                            nn_thresh=0.7,
                            cuda=False)
    print('==> Successfully loaded pre-trained network.')

    # This class helps merge consecutive point matches into tracks.
    max_length = len(raw_data.timestamps) // args.n_skip + 1
    tracker = sp.PointTracker(max_length=max_length, nn_thresh=fe.nn_thresh)

    print('==> Running SuperPoint')
    N = len(raw_data.timestamps)
    idx = range(0, N, args.n_skip);
    for i in idx:
        img, _ = raw_data.get_gray(i) # only get image from cam0
        img_np = np.array(img).astype('float32') / 255.0;
        pts, desc, _ = fe.run(img_np)
        tracker.update(pts, desc)

    vision_data = get_vision_data(tracker);


    """
    Setup GTSAM factor graph with imu measurements and keypoints
    """

    """
    Solve factor graph
    """

    """
    Visualize results
    """






