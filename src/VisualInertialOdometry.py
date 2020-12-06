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

    def add_keypoints(self,vision_data,measured_poses,n_skip):
      K = gtsam.Cal3_S2(984.2439, 980.8141, 0.0, 690.0, 233.1966)
      print(K)
      #print(K[0,0])
      K_np = np.array([[984.244, 0., 690.],[0.,980.814,233.197],[0.,0.,1.]])
      inv_K = np.linalg.inv(K_np)

      #K = gtsam.Cal3_S2(calib_data)
      offset_pose_key = X(0)
      
      measurement_noise = gtsam.noiseModel.Isotropic.Sigma(
        2, 100.0)  # one pixel in u and v
      # measurement_noise = gtsam.noiseModel.Isotropic.Sigma(
      #   noise_data)
      for i in range(vision_data.shape[0]):
        key_point_initialized=False 
        
        #isam.update(graph, initial_estimate)

        for j in range(1000,vision_data.shape[1]):
          if vision_data[i,j,0] >= 0:
            self.graph.push_back(gtsam.GenericProjectionFactorCal3_S2(
              vision_data[i,j,:], measurement_noise, X(j), L(i), K))
            if not key_point_initialized:
              initial_lj = 5.*inv_K@ np.array(
                 [vision_data[i,j,0],vision_data[i,j,1],1])
              #initial_lj = np.array([1.,1.,1.])
              initial_lj = (measured_poses[j*n_skip])@ np.hstack((initial_lj, [1.]))
              self.initial_estimate.insert(L(i), initial_lj[0:3])
              key_point_initialized = True

      #print(self.initial_estimate)
      #print(self.graph)

    def estimate(self, SOLVER_PARAMS=None):
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate, SOLVER_PARAMS)
        self.result = self.optimizer.optimize()

        return self.result



        













