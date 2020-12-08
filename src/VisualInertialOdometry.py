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
import matplotlib.pyplot as plt
np.random.seed(0)

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

    def add_imu_measurements(self, measured_poses, measured_acc, measured_omega, measured_vel, delta_t, n_skip, initial_poses=None):

        n_frames = measured_acc.shape[0]
#       n_frames = measured_poses.shape[0]


        # Check if sizes are correct
#       assert measured_poses.shape[0] == n_frames
#       assert measured_acc.shape[0] == n_frames
#       assert measured_vel.shape[0] == n_frames

        # Pose prior
        pose_key = X(0)
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]))
#       pose_0 = gtsam.Pose3(measured_poses[0])
        pose_0 = gtsam.Pose3(np.eye(4))
        self.graph.push_back(gtsam.PriorFactorPose3(pose_key, pose_0, pose_noise))

        self.initial_estimate.insert(pose_key, gtsam.Pose3(measured_poses[0]))

        # IMU prior
        bias_key = B(0)
        bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.5)
        self.graph.push_back(gtsam.PriorFactorConstantBias(bias_key, gtsam.imuBias.ConstantBias(), bias_noise))

        self.initial_estimate.insert(bias_key, gtsam.imuBias.ConstantBias())

        # Velocity prior
        velocity_key = V(0)
        velocity_noise = gtsam.noiseModel.Isotropic.Sigma(3, .5)
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
                """ Using GT + noise as initialization """
#               DELTA = gtsam.Pose3(gtsam.Rot3.Rodrigues(0, 0, 0.1 * np.random.randn()),
#                                   gtsam.Point3(4 * np.random.randn(), 4 * np.random.randn(), 4 * np.random.randn()))
#               self.initial_estimate.insert(pose_key, gtsam.Pose3(measured_poses[i]).compose(DELTA))
                """ Using cv pose estimate """
                self.initial_estimate.insert(pose_key, gtsam.Pose3(measured_poses[i // n_skip]))

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

    def add_keypoints(self,vision_data,measured_poses,n_skip, depth, axs):
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
      CAM_TO_IMU_POSE = gtsam.Pose3(cam_to_imu)
      imu_to_cam = np.linalg.inv(cam_to_imu)
      IMU_TO_CAM_POSE = gtsam.Pose3(imu_to_cam)
      print("IMU_TO_CAM_POSE", IMU_TO_CAM_POSE)

      K_np = np.array([[9.895267e+02, 0.000000e+00, 7.020000e+02], 
                       [0.000000e+00, 9.878386e+02, 2.455590e+02], 
                       [0.000000e+00, 0.000000e+00, 1.000000e+00]]) 
#     K_np = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02], 
#                      [0.000000e+00, 7.215377e+02, 1.728540e+02], 
#                      [0.000000e+00, 0.000000e+00, 1.000000e+00]]) 

      K = gtsam.Cal3_S2(K_np[0,0], K_np[1,1], 0., K_np[0,2], K_np[1,2])
      print("K_np = ", K_np)


      valid_track = np.zeros(vision_data.shape[0], dtype=bool)
      N = vision_data.shape[1]
      for i in range(vision_data.shape[0]):
          track_length = N - np.sum(vision_data[i,:,0] == -1)
          if track_length > 1 and track_length < 0.5*N:
              valid_track[i] = True
      print('valid tracks: ')
      print(valid_track)


      count = 0
      measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 10.0) 
      for i in range(0, vision_data.shape[0], 20):
        if not valid_track[i]:
            continue
        key_point_initialized=False 
        for j in range(vision_data.shape[1]-1):
          if vision_data[i,j,0] >= 0:
            zp = float(depth[j * n_skip][vision_data[i,j,1], vision_data[i,j,0], 2])
            if zp == 0:
                continue
            self.graph.push_back(gtsam.GenericProjectionFactorCal3_S2(
              vision_data[i,j,:], measurement_noise, X(j), L(i), K, IMU_TO_CAM_POSE))
            if not key_point_initialized:
                count += 1
                """
                Method 1
                """
#             initial_lj = inv_K @ np.array([vision_data[i,j,0],vision_data[i,j,1], 1])
#             # Rotate axes
#             initial_lj = np.array([initial_lj[2], -initial_lj[0], -initial_lj[1]]) + np.array([10.,0,0])
#             # Convert to global
#             initial_lj = measured_poses[j*n_skip] @ np.hstack((initial_lj, [1.]))
#             print(initial_lj)
#             self.initial_estimate.insert(L(i), initial_lj[:3])
                """
                Method 2
                """
#                X1 = inv_K @ np.array([vision_data[i,j,0],vision_data[i,j,1], 1])
#                X1 = np.array([X1[2], -X1[0], -X1[1], 1])
#                X2 = inv_K @ np.array([vision_data[i,j+1,0],vision_data[i,j+1,1], 1])
#                X2 = np.array([X2[2], -X2[0], -X2[1], 1])
#                Xg = np.linalg.pinv(np.vstack((measured_poses[j*n_skip], 
#                                               measured_poses[(j+1)*n_skip]))) 
#                print('Xg.shape = ', Xg.shape)
#                Xg = Xg @ np.hstack((X1, X2))
#                print('Xg.shape = ', Xg.shape)
#                self.initial_estimate.insert(L(i), Xg[:3])
                """
                Method 3
                """
                fx = K_np[0,0]
                fy = K_np[1,1]
                cx = K_np[0,2]
                cy = K_np[1,2]

                # Depth:
#               zp = 10. + 0.5 * np.random.randn() # m
#               zp = 5 
                zp = float(depth[j * n_skip][vision_data[i,j,1], vision_data[i,j,0], 2])
                xp = float(vision_data[i,j,0] - cx) / fx * zp
                yp = float(vision_data[i,j,1] - cy) / fy * zp
                print(xp, yp, zp)

                # Convert to global
#               Xg = measured_poses[j*n_skip] @ imu_to_cam @ np.array([xp, yp, zp, 1])
                Xg = measured_poses[j] @ imu_to_cam @ np.array([xp, yp, zp, 1])
                print(Xg)
                axs.scatter(Xg[0], Xg[1], s=10)
                self.initial_estimate.insert(L(i), Xg[:3])
                
                key_point_initialized = True
      print('===============> Using ', count, ' tracks')

    def estimate(self, SOLVER_PARAMS=None, marginals=False):
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate, SOLVER_PARAMS)
        self.result = self.optimizer.optimize()

#       for i in range(20):
#           self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.result, SOLVER_PARAMS)
#           self.result = self.optimizer.optimize()

        if marginals:
            self.marginals = gtsam.Marginals(self.graph, self.result)
            return self.result, self.marginals
        else:
            return self.result



        













