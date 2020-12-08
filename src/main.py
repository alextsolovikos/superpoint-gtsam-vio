import numpy as np
import VisualInertialOdometry as vio
import pykitti
import argparse
import SuperPointPretrainedNetwork.demo_superpoint as sp
import cv2
import os
from scipy.spatial.transform import Rotation as R

import gtsam
from gtsam.symbol_shorthand import B, V, X, L

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', size=16)

def get_theta(rotation):
    return R.from_matrix(rotation).as_euler('xyz')

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


def estimate_poses(vision_data):
   R_rect = np.array([[9.999239e-01, 9.837760e-03, -7.445048e-03, 0.],
                      [ -9.869795e-03, 9.999421e-01, -4.278459e-03, 0.],
                      [ 7.402527e-03, 4.351614e-03, 9.999631e-01, 0.],
                      [ 0., 0., 0., 1.]])
   R_cam_velo = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04],
                          [ 1.480249e-02, 7.280733e-04, -9.998902e-01],
                          [ 9.998621e-01, 7.523790e-03, 1.480755e-02]])
   R_velo_imu = np.array([[9.999976e-01, 7.553071e-04, -2.035826e-03],
                          [-7.854027e-04, 9.998898e-01, -1.482298e-02],
                          [2.024406e-03, 1.482454e-02, 9.998881e-01]])
   t_cam_velo = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])
   t_velo_imu = np.array([-8.086759e-01, 3.195559e-01, -7.997231e-01])
   T_velo_imu = np.zeros((4,4))
   T_cam_velo = np.zeros((4,4))
   T_velo_imu[3,3] = 1.
   T_cam_velo[3,3] = 1.
   T_velo_imu[:3,:3] = R_velo_imu
   T_velo_imu[:3,3] = t_velo_imu
   T_cam_velo[:3,:3] = R_cam_velo
   T_cam_velo[:3,3] = t_cam_velo
   cam_to_imu = R_rect @ T_cam_velo @ T_velo_imu
   imu_to_cam = np.linalg.inv(cam_to_imu)

   # Estimate list of flu poses using vision data
   s = 10. # scale translations to kind of match GT plot
   A = np.array([(721.5377, 0., 609.5593), (0., 721.5377, 172.8540), (0., 0., 1.)])
   K = np.array([[984.244, 0., 690.],[0.,980.814,233.197],[0.,0.,1.]])
   poses = [np.identity(4)] 
   N = vision_data.shape[1]
   print(N)
   for j in range(1, N):
     pts1 = np.zeros((1,2))
     pts2 = np.zeros((1,2))
     for i in range(vision_data.shape[0]):
         # Collect all point matches between images j-1 and j
         if vision_data[i, j, 0] >= 0 and vision_data[i, j-1, 0] >= 0:
            pts1 = np.vstack((pts1, vision_data[i, j-1]))
            pts2 = np.vstack((pts2, vision_data[i, j]))
     pts1 = np.delete(pts1, 0, 0)
     pts2 = np.delete(pts2, 0, 0)
     pts1 = pts1.astype(float)
     pts2 = pts2.astype(float)
     if pts1.shape[0] > 1:
       E, _ = cv2.findEssentialMat(pts2, pts1, K)
       _, R, t, _ = cv2.recoverPose(E, pts2, pts1, K)
       t = s * t
       rel_pose = np.vstack((np.hstack((R,t)), np.array([0., 0., 0., 1.])))
     else:
       print('Found no matches!') 
       rel_pose = np.identity(4)
     poses.append(poses[j - 1] @ rel_pose)  
   
   # Transform from cam to flu 
   T_cf = np.array([(0, 0, 1, 0), (-1, 0, 0, 0), (0, -1, 0, 0), (0, 0, 0, 1)])
   flu_poses = np.array([T_cf @ p for p in poses])
#  flu_poses = np.array([imu_to_cam @ p for p in poses])


   # Place into gtsam objects
   # print('Converting poses to gtsam objects...')
   # gtsam_poses = np.array([gtsam.Pose3(p) for p in flu_poses])
   # for p in flu_poses:
      # print(p[:3,1])
      # R_g = gtsam.Rot3(gtsam.Point3(p[:3,0]), gtsam.Point3(p[:3,1]), gtsam.Point3(p[:3,2]))
      # print(R_g)
      # t_g = gtsam.Point3(p[3,:3])
      # print(t_g)
      # gtsam_poses.append(gtsam.Pose3(R_g, t_g))
      
   return flu_poses


if __name__ == '__main__':
    # For testing, use the following command in superpoint-gtsam-vio/src:
    #    python3 main.py --basedir data --date '2011_09_26' --drive '0005' --n_skip 10
    parser = argparse.ArgumentParser(description='Visual Inertial Odometry of KITTI dataset.')
    parser.add_argument('--basedir', dest='basedir', type=str)
    parser.add_argument('--date', dest='date', type=str)
    parser.add_argument('--drive', dest='drive', type=str)
    parser.add_argument('--n_skip', dest='n_skip', type=int, default=1)
    parser.add_argument('--n_frames', dest='n_frames', type=int, default=None)
    args = parser.parse_args()

    fig, axs = plt.subplots(1, figsize=(12, 8), facecolor='w', edgecolor='k')
    plt.subplots_adjust(right=0.95, left=0.1, bottom=0.17)

    """ 
    Load KITTI raw data
    """

    data = pykitti.raw(args.basedir, args.date, args.drive)

    # Number of frames
#   n_frames = 
    if args.n_frames is None:
        n_frames = len(data.timestamps)
    else:
        n_frames = args.n_frames

    # Time in seconds
    time = np.array([(data.timestamps[k] - data.timestamps[0]).total_seconds() for k in range(n_frames)])

    # Time step
    delta_t = np.diff(time)

    # Velocity
    measured_vel = np.array([[data.oxts[k][0].vf, data.oxts[k][0].vl, data.oxts[k][0].vu] for k in range(n_frames)])

    # Acceleration
#   measured_acc = np.array([[data.oxts[k][0].ax, data.oxts[k][0].ay, data.oxts[k][0].az] for k in range(n_frames)])
    measured_acc = np.array([[data.oxts[k][0].af, data.oxts[k][0].al, data.oxts[k][0].au] for k in range(n_frames)])

    # Angular velocity
#   measured_omega = np.array([[data.oxts[k][0].wx, data.oxts[k][0].wy, data.oxts[k][0].wz] for k in range(n_frames)])
    measured_omega = np.array([[data.oxts[k][0].wf, data.oxts[k][0].wl, data.oxts[k][0].wu] for k in range(n_frames)])

    # Poses
    measured_poses = np.array([data.oxts[k][1] for k in range(n_frames)])
    measured_poses = np.linalg.inv(measured_poses[0]) @ measured_poses

    """
    Load depth data
    """
    depth_data_path = os.path.join(args.basedir, args.date, '2011_09_26_drive_' + args.drive + '_sync/proj_depth/groundtruth/image_02')
    depth = []

    # Load in the images
    for filepath in sorted(os.listdir(depth_data_path)):
        if filepath[0] == '.':
            continue
        depth.append(cv2.imread(os.path.join(depth_data_path, filepath)))

    """
    Run superpoint to get keypoints
    """
    print('==> Loading pre-trained network.')
    # This class runs the SuperPoint network and processes its outputs.
    # Inputs from list of default options in superpoint_demo.py.
    fe = sp.SuperPointFrontend(weights_path='src/SuperPointPretrainedNetwork/superpoint_v1.pth',
                            nms_dist=4,
                            conf_thresh=0.15,  # 0.015
                            nn_thresh=0.9,
                            cuda=False)
    print('==> Successfully loaded pre-trained network.')

    # This class helps merge consecutive point matches into tracks.
    max_length = n_frames // args.n_skip + 1
    tracker = sp.PointTracker(max_length=max_length, nn_thresh=fe.nn_thresh)

    print('==> Running SuperPoint')
    idx = range(0, n_frames, args.n_skip);
    for i in idx:
        print(i)
        img = data.get_cam1(i) # only get image from cam0
        img_np = np.array(img).astype('float32') / 255.0;
        pts, desc, _ = fe.run(img_np)
        tracker.update(pts, desc)

    print('==> Extracting keypoint tracks')
    vision_data = get_vision_data(tracker);


    """
    Estimate initialization of poses from vision only
    """
    cv_poses = estimate_poses(vision_data)
#   cv_poses = np.linalg.inv(cv_poses[0]) @ cv_poses


    """
    GTSAM parameters
    """
    print('==> Adding IMU factors to graph')

    g = 9.81

    # IMU preintegration parameters
    # Default Params for a Z-up navigation frame, such as ENU: gravity points along negative Z-axis
    IMU_PARAMS = gtsam.PreintegrationParams.MakeSharedU(g)
    I = np.eye(3)
    IMU_PARAMS.setAccelerometerCovariance(I * 0.2)
    IMU_PARAMS.setGyroscopeCovariance(I * 0.2)
    IMU_PARAMS.setIntegrationCovariance(I * 0.2)
#   IMU_PARAMS.setUse2ndOrderCoriolis(False)
#   IMU_PARAMS.setOmegaCoriolis(np.array([0, 0, 0]))

    BIAS_COVARIANCE = gtsam.noiseModel.Isotropic.Variance(6, 0.4)


    """
    Solve IMU-only graph
    """
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(1000)
#   params.setDiagonalDamping(10)
    params.setVerbosity('ERROR')
    params.setVerbosityLM('SUMMARY')
#   params.setVerbosity('SUMMARY')

    print('==> Solving IMU-only graph')
    imu_only = vio.VisualInertialOdometryGraph(IMU_PARAMS=IMU_PARAMS, BIAS_COVARIANCE=BIAS_COVARIANCE)
    imu_only.add_imu_measurements(cv_poses, measured_acc, measured_omega, measured_vel, delta_t, args.n_skip)
#   imu_only.add_imu_measurements(measured_poses, measured_acc, measured_omega, measured_vel, delta_t, args.n_skip)
    result_imu = imu_only.estimate(params)



    """
    Solve VIO graph
    """
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(1000)
    params.setlambdaUpperBound(1.e+6)
    params.setlambdaLowerBound(0.1)
    params.setDiagonalDamping(1000)
    params.setVerbosity('ERROR')
    params.setVerbosityLM('SUMMARY')
    params.setRelativeErrorTol(1.e-9)
    params.setAbsoluteErrorTol(1.e-9)
#   params.setVerbosity('SUMMARY')



    print('==> Solving VIO graph')
    vio_full = vio.VisualInertialOdometryGraph(IMU_PARAMS=IMU_PARAMS, BIAS_COVARIANCE=BIAS_COVARIANCE)
    vio_full.add_imu_measurements(cv_poses, measured_acc, measured_omega, measured_vel, delta_t, args.n_skip)
    vio_full.add_keypoints(vision_data, cv_poses, args.n_skip, depth, axs)
#   vio_full.add_imu_measurements(measured_poses, measured_acc, measured_omega, measured_vel, delta_t, args.n_skip)
#   vio_full.add_keypoints(vision_data, measured_poses, args.n_skip, depth, axs)

    result_full = vio_full.estimate(SOLVER_PARAMS=params)
#   result_full, marginals_full = vio_full.estimate(SOLVER_PARAMS=params, marginals=True)



    """
    Visualize results
    """
    print('==> Plotting results')

    x_gt = measured_poses[:,0,3]
    y_gt = measured_poses[:,1,3]
    theta_gt = np.array([get_theta(measured_poses[k,:3,:3])[2] for k in range(n_frames)])

#   x_init = np.array([vio_full.initial_estimate.atPose3(X(k)).translation()[0] for k in range(n_frames//args.n_skip)]) 
#   y_init = np.array([vio_full.initial_estimate.atPose3(X(k)).translation()[1] for k in range(n_frames//args.n_skip)]) 
    x_init = np.array([vio_full.initial_estimate.atPose3(X(k)).translation()[0] for k in range(n_frames//args.n_skip)]) 
    y_init = np.array([vio_full.initial_estimate.atPose3(X(k)).translation()[1] for k in range(n_frames//args.n_skip)]) 
    theta_init = np.array([get_theta(vio_full.initial_estimate.atPose3(X(k)).rotation().matrix())[2] for k in range(n_frames//args.n_skip)]) 

    x_est_full = np.array([result_full.atPose3(X(k)).translation()[0] for k in range(n_frames//args.n_skip)]) 
    y_est_full = np.array([result_full.atPose3(X(k)).translation()[1] for k in range(n_frames//args.n_skip)]) 
    theta_est_full = np.array([get_theta(result_full.atPose3(X(k)).rotation().matrix())[2] for k in range(n_frames//args.n_skip)]) 

    x_est_imu = np.array([result_imu.atPose3(X(k)).translation()[0] for k in range(n_frames//args.n_skip)]) 
    y_est_imu = np.array([result_imu.atPose3(X(k)).translation()[1] for k in range(n_frames//args.n_skip)]) 
    theta_est_imu = np.array([get_theta(result_imu.atPose3(X(k)).rotation().matrix())[2] for k in range(n_frames//args.n_skip)]) 

    axs.plot(x_gt, y_gt, color='k', label='GT')
    axs.plot(x_init, y_init, 'x-', color='m', label='Initial')
    axs.plot(x_est_imu, y_est_imu, 'o-', color='r', label='IMU')
    axs.plot(x_est_full, y_est_full, 'o-', color='b', label='VIO')
    axs.set_xlabel('$x\ (m)$')
    axs.set_ylabel('$y\ (m)$')
    axs.set_aspect('equal', 'box')
    plt.grid(True)

#   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend()
    plt.savefig('/Users/atsol/autonomous_robots/projects/final_project/path.eps')
    plt.show()

    # Plot pose as time series
    fig, axs = plt.subplots(3, figsize=(8, 8), facecolor='w', edgecolor='k')
    plt.subplots_adjust(right=0.95, left=0.15, bottom=0.17, hspace=0.5)
    # Plot x
    axs[0].grid(True)
    axs[0].plot(time, x_gt, color='k', label='GT')
    axs[0].plot(time[:n_frames-1:args.n_skip], x_init, color='m', label='Initial')
    axs[0].plot(time[:n_frames-1:args.n_skip], x_est_imu, color='r', label='IMU')
    axs[0].plot(time[:n_frames-1:args.n_skip], x_est_full, color='b', label='VIO')
    axs[0].set_xlabel('$t\ (s)$')
    axs[0].set_ylabel('$x\ (m)$')

    # Plot y
    axs[1].grid(True)
    axs[1].plot(time, y_gt, color='k', label='GT')
    axs[1].plot(time[:n_frames-1:args.n_skip], y_init, color='m', label='Initial')
    axs[1].plot(time[:n_frames-1:args.n_skip], y_est_imu, color='r', label='IMU')
    axs[1].plot(time[:n_frames-1:args.n_skip], y_est_full, color='b', label='VIO')
    axs[1].set_xlabel('$t\ (s)$')
    axs[1].set_ylabel('$y\ (m)$')

    # Plot theta
    axs[2].grid(True)
    axs[2].plot(time, theta_gt, color='k', label='GT')
    axs[2].plot(time[:n_frames-1:args.n_skip], theta_init, color='m', label='Initial')
    axs[2].plot(time[:n_frames-1:args.n_skip], theta_est_imu, color='r', label='IMU')
    axs[2].plot(time[:n_frames-1:args.n_skip], theta_est_full, color='b', label='VIO')
    axs[2].set_xlabel('$t\ (s)$')
    axs[2].set_ylabel('$\\theta\ (rad)$')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('/Users/atsol/autonomous_robots/projects/final_project/poses.eps')
    plt.show()

    # Plot pose as time series
    fig, axs = plt.subplots(3, figsize=(8, 8), facecolor='w', edgecolor='k')
    plt.subplots_adjust(right=0.95, left=0.15, bottom=0.17, hspace=0.5)
    # Plot x
    axs[0].grid(True)
    axs[0].plot(time[:n_frames-1:args.n_skip], np.abs(x_gt[:n_frames-1:args.n_skip] - x_init), color='m', label='Initial')
    axs[0].plot(time[:n_frames-1:args.n_skip], np.abs(x_gt[:n_frames-1:args.n_skip] - x_est_imu), color='r', label='IMU')
    axs[0].plot(time[:n_frames-1:args.n_skip], np.abs(x_gt[:n_frames-1:args.n_skip] - x_est_full), color='b', label='VIO')
    axs[0].set_xlabel('$t\ (s)$')
    axs[0].set_ylabel('$e_x\ (m)$')

    # Plot y
    axs[1].grid(True)
    axs[1].plot(time[:n_frames-1:args.n_skip], np.abs(y_gt[:n_frames-1:args.n_skip] - y_init), color='m', label='Initial')
    axs[1].plot(time[:n_frames-1:args.n_skip], np.abs(y_gt[:n_frames-1:args.n_skip] - y_est_imu), color='r', label='IMU')
    axs[1].plot(time[:n_frames-1:args.n_skip], np.abs(y_gt[:n_frames-1:args.n_skip] - y_est_full), color='b', label='VIO')
    axs[1].set_xlabel('$t\ (s)$')
    axs[1].set_ylabel('$e_y\ (m)$')

    # Plot theta
    axs[2].grid(True)
    axs[2].plot(time[:n_frames-1:args.n_skip], np.abs(theta_gt[:n_frames-1:args.n_skip] - theta_init), color='m', label='Initial')
    axs[2].plot(time[:n_frames-1:args.n_skip], np.abs(theta_gt[:n_frames-1:args.n_skip] - theta_est_imu), color='r', label='IMU')
    axs[2].plot(time[:n_frames-1:args.n_skip], np.abs(theta_gt[:n_frames-1:args.n_skip] - theta_est_full), color='b', label='VIO')
    axs[2].set_xlabel('$t\ (s)$')
    axs[2].set_ylabel('$e_{\theta}\ (rad)$')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('/Users/atsol/autonomous_robots/projects/final_project/errors.eps')
    plt.show()

#   # Print vision_data matrix
#   track_exists = np.zeros_like(vision_data[:,:,0])
#   track_exists[track_exists != -1] = 1
#   plt.imshow(track_exists, aspect='auto')
#   plt.show()
    


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




