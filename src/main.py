import VisualInertialOdometry as vio
import pykitti
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visual Inertial Odometry of KITTI dataset.')
    parser.add_argument('--basedir', dest='basedir', type=str)
    parser.add_argument('--date', dest='date', type=str)
    parser.add_argument('--drive', dest='drive', type=str)
    args = parser.parse_args()

    raw_data = pykitti.raw(args.basedir, args.date, args.drive)


    """
    Run superpoint to get keypoints
    """

    """
    Setup GTSAM factor graph with imu measurements and keypoints
    """

    """
    Solve factor graph
    """

    """
    Visualize results
    """













