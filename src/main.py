import VisualInertialOdometry as vio
import pykitti
import matplotlib.pyplot as plt
import argparse
import numpy as np
plt.rc('text', usetex=True)
plt.rc('font', size=16)

import gtsam
from gtsam.symbol_shorthand import B, V, X, L


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visual Inertial Odometry of KITTI dataset.')
    parser.add_argument('--basedir', dest='basedir', type=str)
    parser.add_argument('--date', dest='date', type=str)
    parser.add_argument('--drive', dest='drive', type=str)
    args = parser.parse_args()

    data = pykitti.raw(args.basedir, args.date, args.drive)

    # Number of frames
    n_frames = len(data.timestamps)

    # Time in seconds
    time = np.array([(data.timestamps[k] - data.timestamps[0]).total_seconds() for k in range(n_frames)])

    # Time step
    delta_t = np.diff(time)

    # Velocity
    measured_vel = np.array([[data.oxts[k][0][8], data.oxts[k][0][9], data.oxts[k][0][10]] for k in range(n_frames)])

    # Acceleration
#   measured_acc_2 = np.array([[data.oxts[k][0][14], data.oxts[k][0][15], data.oxts[k][0][16]] for k in range(n_frames)])
    measured_acc = np.array([[data.oxts[k][0][11], data.oxts[k][0][12], data.oxts[k][0][13]] for k in range(n_frames)])

    # Angular velocity
#   measured_omega_2 = np.array([[data.oxts[k][0][20], data.oxts[k][0][21], data.oxts[k][0][22]] for k in range(n_frames)])
    measured_omega = np.array([[data.oxts[k][0][17], data.oxts[k][0][18], data.oxts[k][0][19]] for k in range(n_frames)])

    # Poses
    measured_poses = np.array([data.oxts[k][1] for k in range(n_frames)])

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
        PARAMS.setOmegaCoriolis(np.array([0, 0, 0]))

        BIAS_COVARIANCE = gtsam.noiseModel.Isotropic.Variance(6, 0.5)
        DELTA = gtsam.Pose3(gtsam.Rot3.Rodrigues(0, 0, 0),
                            gtsam.Point3(0.0, 0.0, 0))
        CORR = gtsam.Pose3(gtsam.Rot3.Rodrigues(np.pi, 0, 0),
                           gtsam.Point3(0, 0, 0))

        return PARAMS, BIAS_COVARIANCE, DELTA, CORR

    PARAMS, BIAS_COVARIANCE, DELTA, CORR = preintegration_parameters(g)

    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # Pose x0 prior
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5, 0.5, 0.5, 0.6, 0.6, 0.6]))
    pose_0 = gtsam.Pose3(measured_poses[0])
    graph.push_back(gtsam.PriorFactorPose3(X(0), pose_0, pose_noise))

    initial_estimate.insert(X(0), gtsam.Pose3(measured_poses[0]))

    # IMU prior
    bias_key = B(0)
    bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.5)
    graph.push_back(gtsam.PriorFactorConstantBias(bias_key, gtsam.imuBias.ConstantBias(), bias_noise))

    initial_estimate.insert(bias_key, gtsam.imuBias.ConstantBias())

    # Velocity prior
    velocity_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.5)
    velocity_0 = measured_vel[0]
    graph.push_back(gtsam.PriorFactorVector(V(0), velocity_0, velocity_noise))

    initial_estimate.insert(V(0), velocity_0)
    

    # Preintegrator
    accum = gtsam.PreintegratedImuMeasurements(PARAMS)

    # Add measurements to factor graph
    for i in range(n_frames):
        if i > 0:
            initial_estimate.insert(X(i), gtsam.Pose3(measured_poses[i]).compose(DELTA))

            if i % 5 == 0:
                bias_key += 1
                graph.add(gtsam.BetweenFactorConstantBias(bias_key - 1, bias_key, gtsam.imuBias.ConstantBias(), BIAS_COVARIANCE))
                initial_estimate.insert(bias_key, gtsam.imuBias.ConstantBias())

            accum.integrateMeasurement(measured_acc[i], measured_omega[i], delta_t[i-1])

            # Add IMU Factor
            graph.add(gtsam.ImuFactor(X(i - 1), V(i - 1), X(i), V(i), bias_key, accum))

            # Insert velocity 
            initial_estimate.insert(V(i), measured_vel[i])
            accum.resetIntegration()




    """
    Solve factor graph
    """

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(1000)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()





    """
    Visualize results
    """

    fig, axs = plt.subplots(1, figsize=(8, 6), facecolor='w', edgecolor='k')
    plt.subplots_adjust(right=0.95, left=0.1, bottom=0.17)

    x_gt = measured_poses[:,0,3]
    y_gt = measured_poses[:,1,3]

    x_est = np.array([result.atPose3(X(k)).translation()[0] for k in range(n_frames)]) 
    y_est = np.array([result.atPose3(X(k)).translation()[1] for k in range(n_frames)]) 

    axs.plot(x_gt, y_gt, color='k')
    axs.plot(x_est, y_est, color='b')
    axs.set_aspect('equal', 'box')

    plt.show()



    # Compare xyz with flu frame measurements
#   fig, axs = plt.subplots(3, figsize=(10, 3.5), facecolor='w', edgecolor='k')
#   plt.subplots_adjust(right=0.95, left=0.1, bottom=0.17)

#   axs[0].plot(time, measured_acc[:,0], color='k')
#   axs[0].plot(time, measured_acc_2[:,0], color='b')
#   axs[1].plot(time, measured_acc[:,1], color='k')
#   axs[1].plot(time, measured_acc_2[:,1], color='b')
#   axs[2].plot(time, measured_acc[:,2], color='k')
#   axs[2].plot(time, measured_acc_2[:,2], color='b')

#   plt.show()






