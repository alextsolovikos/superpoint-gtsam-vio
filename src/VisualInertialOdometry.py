"""
The University of Texas at Austin
CS393R Fall 2020
Space Robots Team
Final Project

Authors: Kristen Michaelson, Alex Tsolovikos, Enrico Zucchelli

Description: This is a class that implements a GTSAM factor graph for Visual Inertial Odometry (VIO)
"""




import cv2
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

        n_frames = measured_acc.shape[0]

        # Check if sizes are correct
        # assert measured_poses.shape[0] == n_frames
        # assert measured_acc.shape[0] == n_frames
        assert measured_vel.shape[0] == n_frames

        # Pose prior
        pose_key = X(0)
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.4, 0.4, 0.4, 0.2, 0.2, 0.2]))
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

    def estimate_poses(self, vision_data):
       s = 2.2 # lazy; scale translations to kind of match GT plot
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

       return poses

    """
    def estimate_flu_poses(self, cam_poses): 
       # Transform from cam to flu 
       T_cf = np.array([(0, 0, 1, 0), (-1, 0, 0, 0), (0, -1, 0, 0), (0, 0, 0, 1)])
       flu_poses = [T_cf @ cp for cp in cam_poses]
       return flu_poses
    """

    def add_keypoints(self,vision_data,measured_poses,n_skip):
      print('Adding keypoints...')
      K = gtsam.Cal3_S2(984.2439, 980.8141, 0.0, 690.0, 233.1966)
      A = gtsam.Cal3_S2(721.5377, 721.5377, 0.0, 609.5593, 172.8540)
      # T_cf = np.array([(0, 0, 1, 0), (-1, 0, 0, 0), (0, -1, 0, 0), (0, 0, 0, 1)])
      # T_cf = gtsam.Pose3(gtsam.Rot3(0., 0., 1., -1., 0., 0., 0., -1., 0.),
      #                   gtsam.Point3(0., 0., 0.))
      T_cf = gtsam.Pose3(gtsam.Rot3(0., -1., 0., 0., 0., -1., 1., 0., 0.),
                         gtsam.Point3(0., 0., 0.))
      print(K)
      #print(K[0,0])
      A_np = np.array([(721.5377, 0., 609.5593), (0., 721.5377, 172.8540), (0., 0., 1.)])
      K_np = np.array([[984.244, 0., 690.],[0.,980.814,233.197],[0.,0.,1.]])
      inv_K = np.linalg.inv(K_np)

      #K = gtsam.Cal3_S2(calib_data)
      offset_pose_key = X(0)
      
      measurement_noise = gtsam.noiseModel.Isotropic.Sigma(
        2, 2.0)  # 2 pixel in u and v
      # measurement_noise = gtsam.noiseModel.Isotropic.Sigma(
      #   noise_data)
      # estimated_poses = self.estimate_flu_poses(self.estimate_poses(vision_data)) # lazy
      for i in range(vision_data.shape[0]):
        key_point_initialized=False 
                

        for j in range(vision_data.shape[1]):
          if vision_data[i,j,0] >= 0:
            self.graph.push_back(gtsam.GenericProjectionFactorCal3_S2(
              vision_data[i,j,:], measurement_noise, X(j), L(i), A, T_cf))
            if not key_point_initialized:
              # initial_lj = 5.*inv_K@ np.array(
              #   [vision_data[i,j,0],vision_data[i,j,1],1])
              initial_lj = measured_poses[j] @ np.array([0., 0., 5., 1])
              # initial_lj = (measured_poses[j*n_skip])@ np.hstack((initial_lj, [1.]))
              self.initial_estimate.insert(L(i), initial_lj[:3])
              print(initial_lj[:3])
              # self.initial_estimate.insert(L(i), estimated_poses[j][0:3,3]) # inside the camera
              key_point_initialized = True

      #print(self.initial_estimate)
      #print(self.graph)
         
             


    def estimate(self, SOLVER_PARAMS=None):
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate, SOLVER_PARAMS)
        self.result = self.optimizer.optimize()

        return self.result



        













