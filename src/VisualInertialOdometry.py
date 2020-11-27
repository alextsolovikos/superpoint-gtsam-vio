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
    
    def __init__(self, camera_noise)
        """
        Define factor graph parameters (e.g. noise, camera calibrations, etc) here
        """

    def add_keypoints(self, ...):

        # do stuff

    def add_imu_measurements(self, ...):

        # do stuff

    def estimate(self):

        # Solve graph and estimate poses


        













