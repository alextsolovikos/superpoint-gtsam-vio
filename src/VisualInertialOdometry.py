"""
The University of Texas at Austin
CS393R Fall 2020
Space Robots Team
Final Project

Authors: Kristen Michaelson, Alex Tsolovikos, Enrico Zucchelli

Description: This is a class that implements a GTSAM factor graph for Visual Inertial Odometry (VIO)
"""





import numpy as np
import gtsam
from gtsam.symbol_shorthand import B, V, X, L

"""
USEFUL CLASSES
    gtsam.PreintegratedImuMeasurements
    gtsam.imuBias.*
    gtsam.noiseModel.Diagonal.Sigmas()
    gtsam.GenericProjectionFactorCal3_S2()
    gtsam.noiseModel.Isotropic.Sigma()
    from gtsam.symbol_shorthand import X, L: we can use X for the poses and L for the keypoints


    


"""

class VisualInertialOdometryGraph(object):
    
    def __init__(self, IMU_PARAMS=None, BIAS_COVARIANCE=None):
        """
        Define factor graph parameters (e.g. noise, camera calibrations, etc) here
        """
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.IMU_PARAMS = IMU_PARAMS
        self.BIAS_COVARIANCE = BIAS_COVARIANCE
        DELTA = gtsam.Pose3(gtsam.Rot3.Rodrigues(0, 0, 0.5),
                            gtsam.Point3(2.0, -2.0, 0.5))

    def add_imu_measurements(self, measured_poses, measured_acc, measured_omega, measured_vel, delta_t, n_skip):

        n_frames = measured_poses.shape[0]

        # Check if sizes are correct
        assert measured_poses.shape[0] == n_frames
        assert measured_acc.shape[0] == n_frames
        assert measured_vel.shape[0] == n_frames

        # Pose prior
        pose_key = X(0)
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0., 0., 0., 0., 0., 0.]))
        pose_0 = gtsam.Pose3(measured_poses[0])
        self.graph.push_back(gtsam.PriorFactorPose3(pose_key, pose_0, pose_noise))

        self.initial_estimate.insert(pose_key, gtsam.Pose3(measured_poses[0]))

        # IMU prior
        bias_key = B(0)
        bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.5)
        self.graph.push_back(gtsam.PriorFactorConstantBias(bias_key, gtsam.imuBias.ConstantBias(), bias_noise))

        self.initial_estimate.insert(bias_key, gtsam.imuBias.ConstantBias())

        # Velocity prior
        velocity_key = V(0)
        velocity_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.5)
        velocity_0 = measured_vel[0]
        self.graph.push_back(gtsam.PriorFactorVector(velocity_key, velocity_0, velocity_noise))

        self.initial_estimate.insert(velocity_key, velocity_0)
        

        # Preintegrator
        accum = gtsam.PreintegratedImuMeasurements(self.IMU_PARAMS)

        # Add measurements to factor graph
        for i in range(1, n_frames):
            accum.integrateMeasurement(measured_acc[i], measured_omega[i], delta_t[i-1])
            if i % n_skip == 0:
                pose_key += 1
                self.initial_estimate.insert(pose_key, gtsam.Pose3(measured_poses[i]))

                velocity_key += 1
                self.initial_estimate.insert(velocity_key, measured_vel[i])

                bias_key += 1
                self.graph.add(gtsam.BetweenFactorConstantBias(bias_key - 1, bias_key, gtsam.imuBias.ConstantBias(), self.BIAS_COVARIANCE))
                self.initial_estimate.insert(bias_key, gtsam.imuBias.ConstantBias())

                # Add IMU Factor
                self.graph.add(gtsam.ImuFactor(pose_key - 1, velocity_key - 1, pose_key, velocity_key, bias_key, accum))

                # Reset preintegration
                accum.resetIntegration()


        # do stuff

    def add_keypoints(self, vision_data, init_guess_poses, n_skip,use_imu=True):
      identity_pose = np.eye(4)
      K = gtsam.Cal3_S2(984.2439, 980.8141, 0.0, 690.0, 233.1966)
      K = gtsam.Cal3_S2(721.5377, 721.5377, 0.0, 609.5593,172.854 )
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
      T_velo_imu[0:3,0:3] = R_velo_imu
      T_velo_imu[0:3,3] = t_velo_imu
      T_cam_velo[0:3,0:3] = R_cam_velo
      T_cam_velo[0:3,3] = t_cam_velo
      cam_to_imu = R_rect.dot(T_cam_velo.dot(T_velo_imu))
      print(np.linalg.det(cam_to_imu))
      imu_to_cam = np.linalg.inv(cam_to_imu)
      print(np.linalg.det(imu_to_cam))
      print(imu_to_cam)
      print(cam_to_imu)
      print(K)
      #print(K[0,0])
      K_np = np.array([[984.244, 0., 690.],[0.,980.814,233.197],[0.,0.,1.]])
      K_np = np.array([[721.5377, 0., 609.5593],[0.,721.5377,172.854],[0.,0.,1.]])
      inv_K = np.linalg.inv(K_np)

      cam2imu_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0., 0., 0., 0., 0., 0.]))
      #K = gtsam.Cal3_S2(calib_data)
      
      measurement_noise = gtsam.noiseModel.Isotropic.Sigma(
        2, 1.0)  # one pixel in u and v
      # measurement_noise = gtsam.noiseModel.Isotropic.Sigma(
      #   noise_data)
      #from mpl_toolkits.mplot3d import Axes3D
      import matplotlib.pyplot as plt
      #fig = plt.figure()
      #ax = fig.add_subplot(111, projection='3d')
      n_keypoints = 0
      print(vision_data.shape)
      print(init_guess_poses.shape)
      for i in range(vision_data.shape[0]):
        key_point_initialized=False 
        

        count = 0.
        for j in range(vision_data.shape[1]):
          if i == 0:
            if not use_imu:
              if j == 0:
                pose0_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0., 0., 0., 0., 0., 0.]))
                self.graph.push_back(gtsam.PriorFactorPose3(X(0),
                  gtsam.Pose3(init_guess_poses[0]), pose0_noise))
              self.initial_estimate.insert(
                X(j), gtsam.Pose3(init_guess_poses[j*n_skip]))
            self.initial_estimate.insert(
              X(j+vision_data.shape[1]), gtsam.Pose3(
              init_guess_poses[j*n_skip]))
            self.graph.push_back(gtsam.BetweenFactorPose3(
              X(j+vision_data.shape[1]), X(j), gtsam.Pose3(
              cam_to_imu), cam2imu_noise))
          if vision_data[i,j,0] >= 0:
            count += 1.
        if count/float(vision_data.shape[1]) < .3 and count >= 1.9:
          n_keypoints += 1

          for j in range(vision_data.shape[1]):
            if vision_data[i,j,0] >= 0:
              self.graph.push_back(gtsam.GenericProjectionFactorCal3_S2(
                vision_data[i,j,:], measurement_noise,
                X(j+vision_data.shape[1]), L(i), K))
              if not key_point_initialized:
                initial_lj = 3.*inv_K@ np.array(
                   [vision_data[i,j,0],vision_data[i,j,1],1])
                initial_lj = np.array([[0.,0.,1.],[0.,1,0],[1,0,0]]).dot(initial_lj)
             #   print(vision_data[i,j])
             #   print(initial_lj)
             #   #initial_lj = np.array([5.,0.,0.])
                initial_lj = (init_guess_poses[j*n_skip])@ np.hstack((initial_lj, [1.]))
                plt.scatter(initial_lj[0],initial_lj[1])
             #   print(i,j)
                print(initial_lj)
             #   print(init_guess_poses[j*n_skip])
             #   #print(inv_K@np.array([0,100,1]))
             #   #print(inv_K@np.array([1350,100,1]))
             #  
             #   print(' ')
                self.initial_estimate.insert(L(i), initial_lj[0:3])
                key_point_initialized = True

      print('Used '+str(n_keypoints)+' out of '+str(vision_data.shape[0]))

    def estimate(self, SOLVER_PARAMS=None):
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate, SOLVER_PARAMS)
        self.result = self.optimizer.optimize()

        return self.result



        













