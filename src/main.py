import VisualInertialOdometry as vio
import pykitti
import matplotlib.pyplot as plt
import argparse
import numpy as np

import gtsam
from gtsam.symbol_shorthand import B, V, X, L


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visual Inertial Odometry of KITTI dataset.')
    parser.add_argument('--basedir', dest='basedir', type=str)
    parser.add_argument('--date', dest='date', type=str)
    parser.add_argument('--drive', dest='drive', type=str)
    args = parser.parse_args()

    raw_data = pykitti.raw(args.basedir, args.date, args.drive)

    # Time in seconds
    time = np.array([(raw_data.timestamps[k] - raw_data.timestamps[0]).total_seconds() for k in range(len(raw_data.timestamps))])
    # Time step
    delta_t = np.diff(time)
    # Acceleration
    measured_acc = np.array([[raw_data.oxts[k][0][11], raw_data.oxts[k][0][12], raw_data.oxts[k][0][13]] for k in range(len(raw_data.oxts))])
    # Angular velocity
    measured_omega = np.array([[raw_data.oxts[k][0][17], raw_data.oxts[k][0][18], raw_data.oxts[k][0][19]] for k in range(len(raw_data.oxts))])

    """
    Run superpoint to get keypoints
    """

    """
    Setup GTSAM factor graph with imu measurements and keypoints
    """

    g = 9.81

    def preintegration_parameters(g):
        # IMU preintegration parameters
        # Default Params for a Z-up navigation frame, such as ENU: gravity points along negative Z-axis
        PARAMS = gtsam.PreintegrationParams.MakeSharedU(g)
        I = np.eye(3)
        PARAMS.setAccelerometerCovariance(I * 0.1)
        PARAMS.setGyroscopeCovariance(I * 0.1)
        PARAMS.setIntegrationCovariance(I * 0.1)
        PARAMS.setUse2ndOrderCoriolis(False)
        PARAMS.setOmegaCoriolis(vector3(0, 0, 0))

        BIAS_COVARIANCE = gtsam.noiseModel.Isotropic.Variance(6, 0.1)
        DELTA = Pose3(Rot3.Rodrigues(0, 0, 0),
                      Point3(0.05, -0.10, 0.20))

        return PARAMS, BIAS_COVARIANCE, DELTA

    PARAMS, BIAS_COVARIANCE, DELTA = preintegration_parameters(g)

    graph = NonlinearFactorGraph()




    """
    Solve factor graph
    """

    """
    Visualize results
    """













