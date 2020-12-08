import numpy as np
import VisualInertialOdometry as vio
import pykitti
import argparse
import SuperPointPretrainedNetwork.demo_superpoint as sp
import cv2

import gtsam
from gtsam.symbol_shorthand import B, V, X, L

import matplotlib.pyplot as plt
#plt.rc('text', usetex=True)
plt.rc('font', size=16)

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
   # Estimate list of flu poses using vision data
   s = 5. # scale translations to kind of match GT plot
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
       E, _ = cv2.findEssentialMat(pts2, pts1, A)
       _, R, t, _ = cv2.recoverPose(E, pts2, pts1, A)
       t = s * t
       rel_pose = np.vstack((np.hstack((R,t)), np.array([0., 0., 0., 1.])))
     else:
       print('Found no matches!') 
       rel_pose = np.identity(4)
     poses.append(poses[j - 1] @ rel_pose)  
   
   # Transform from cam to flu 
   T_cf = np.array([(0, 0, 1, 0), (-1, 0, 0, 0), (0, -1, 0, 0), (0, 0, 0, 1)])
   flu_poses = [T_cf @ p for p in poses]

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
    args = parser.parse_args()

    """ 
    Load KITTI raw data
    """

    data = pykitti.raw(args.basedir, args.date, args.drive)

    # Number of frames
    n_frames = len(data.timestamps)

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
    max_length = len(data.timestamps) // args.n_skip + 1
    tracker = sp.PointTracker(max_length=max_length, nn_thresh=fe.nn_thresh)

    print('==> Running SuperPoint')
    N = len(data.timestamps)
    idx = range(0, N, args.n_skip);
    for i in idx:
        img, _ = data.get_gray(i) # only get image from cam0
        img_np = np.array(img).astype('float32') / 255.0;
        pts, desc, _ = fe.run(img_np)
        tracker.update(pts, desc)

    print('==> Extracting keypoint tracks')
    vision_data = get_vision_data(tracker);
    print(vision_data[3,:,:])
    print(vision_data.shape)


    """
    Add IMU factors
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

    cv_poses = estimate_poses(vision_data)

#    vio_full = vio.VisualInertialOdometryGraph(IMU_PARAMS=IMU_PARAMS, BIAS_COVARIANCE=BIAS_COVARIANCE)
#    vio_full.add_imu_measurements(measured_poses, measured_acc, measured_omega, measured_vel, delta_t, args.n_skip)
#    vio_full.add_keypoints(vision_data, measured_poses, args.n_skip)
#    imu_only = vio.VisualInertialOdometryGraph(IMU_PARAMS=IMU_PARAMS, BIAS_COVARIANCE=BIAS_COVARIANCE)
#    imu_only.add_imu_measurements(measured_poses, measured_acc, measured_omega, measured_vel, delta_t, args.n_skip)
 
    vio_full = vio.VisualInertialOdometryGraph(IMU_PARAMS=IMU_PARAMS, BIAS_COVARIANCE=BIAS_COVARIANCE)
    vio_full.add_imu_measurements(cv_poses, measured_acc, measured_omega, measured_vel, delta_t, args.n_skip)
    vio_full.add_keypoints(vision_data, cv_poses, args.n_skip)
    imu_only = vio.VisualInertialOdometryGraph(IMU_PARAMS=IMU_PARAMS, BIAS_COVARIANCE=BIAS_COVARIANCE)
    imu_only.add_imu_measurements(cv_poses, measured_acc, measured_omega, measured_vel, delta_t, args.n_skip)

    """
    Add Vision factors
    """
    print('==> Adding Vision factors to graph')


    """
    Solve factor graph
    """
    # print('==> Solving factor graph')

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(1)
    params.setlambdaUpperBound(1e12)
    params.setVerbosity('ERROR')
    print('==> Solving full VIO factor graph')
    result_full = vio_full.estimate(params)
    print('==> Solving IMU factor graph')
    result_imu = imu_only.estimate(params)

    """
    Visualize results
    """
    print('==> Plotting results')

    fig, axs = plt.subplots(1, figsize=(8, 6), facecolor='w', edgecolor='k')
    plt.subplots_adjust(right=0.95, left=0.1, bottom=0.17)

    x_gt = measured_poses[:,0,3]
    y_gt = measured_poses[:,1,3]

    """    
    K_np = np.array([[984.244, 0., 690.],[0.,980.814,233.197],[0.,0.,1.]])
    cam_poses_init = vio_full.estimate_poses(vision_data)
    flu_poses_init = vio_full.estimate_flu_poses(cam_poses_init)
    x_init = np.array([pose[0,3] for pose in flu_poses_init]) 
    y_init = np.array([pose[1,3] for pose in flu_poses_init]) 
    """
    gtsam_cv_poses = np.array([gtsam.Pose3(p) for p in cv_poses])
    x_init = np.array([gtsam_cv_poses[k].translation()[0] for k in range(n_frames // args.n_skip)])
    y_init = np.array([gtsam_cv_poses[k].translation()[1] for k in range(n_frames // args.n_skip)])

    x_est_full = np.array([result_full.atPose3(X(k)).translation()[0] for k in range(n_frames//args.n_skip)]) 
    y_est_full = np.array([result_full.atPose3(X(k)).translation()[1] for k in range(n_frames//args.n_skip)]) 


    x_est_imu = np.array([result_imu.atPose3(X(k)).translation()[0] for k in range(n_frames//args.n_skip)]) 
    y_est_imu = np.array([result_imu.atPose3(X(k)).translation()[1] for k in range(n_frames//args.n_skip)]) 

    axs.plot(x_init, y_init, 'o-', color='m', label='INIT')
    axs.plot(x_gt, y_gt, color='k', label='GT')
    axs.plot(x_est_full, y_est_full, 'o-', color='b', label='VIO', linestyle='dashed')
    axs.plot(x_est_imu, y_est_imu, 'o-', color='r', label='IMU')
    axs.set_aspect('equal', 'box')
    plt.legend()
    plt.show()

    
    # Print vision_data matrix
    track_exists = np.zeros_as(vision_data[:,:,0])
    track_exists[track_exists != -1] = 1
    plt.imshow(track_exists, aspect='auto')
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




