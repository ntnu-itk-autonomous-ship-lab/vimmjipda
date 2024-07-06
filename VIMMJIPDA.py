import pathlib
import sys

colav_simulator_path = pathlib.Path(__file__).resolve().parents[2] / "colav_simulator"
sys.path.append(str(colav_simulator_path))

import numpy as np
import scipy.linalg as la
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple


#Colav-simulator imports
from colav_simulator.core.tracking.trackers import ITracker, VIMMJIPDAParams
from colav_simulator.core.sensing import Radar

#VIMMJIPDA Imports
from VIMMJIPDA_interface.VIMMJIPDA.code.tracking.constructs import State, Measurement
from VIMMJIPDA_interface.VIMMJIPDA.code.run import setup_manager
from VIMMJIPDA_interface.VIMMJIPDA.code.tracking.managers import Manager




@dataclass
class VIMMJIPDAParams:
    """Class for holding VIMMJIPDA parameters."""
    IMM_off : bool = field(default_factory=lambda: False)
    single_target : bool = field(default_factory=lambda: False)
    visibility_off : bool = field(default_factory=lambda: False)

    def to_dict(self):
        output_dict = {"IMM_off": self.IMM_off, "single_target": self.single_target, "visibility_off": self.visibility_off}
        return output_dict
    
    @classmethod
    def from_dict(cls, config_dict):
        return VIMMJIPDAParams(IMM_off=config_dict["IMM_off"], single_target=config_dict["single_target"], visibility_off=config_dict["visibility_off"])





class VIMMJIPDA(ITracker):
    """The VIMMJIPDA class implements the VIMMJIPDA (Visibility Interacting Multiple Models Joint Integrated Probabilistic Data Association) tracker by
    Audun Gulliksstad Hem, Edmund Førland Brekke and Lars-Christian Ness Tokle, introduced in the article:
    "Multitarget Tracking With Multiple Models and Visibility: Derivation and Verification on Maritime Radar Data"

    NOTE: It is possible to configure this tracker by turning of the functionality for: Interacting Multiple Models, Visibility and Multi-Target turning
    the tracker into an IPDA tracker
    """

    def __init__(self, sensor_list: list, params: Optional[VIMMJIPDAParams] = None) -> None:
        #TODO Add functionality to input sensors. (Give error message for not using radar???)

        if sensor_list is None:
            raise ValueError("Sensor list must be provided.")


        if params is not None:
            self._params: VIMMJIPDAParams = params
        else:
            self._params = VIMMJIPDAParams()


        self.sensors: list = sensor_list

        self._track_initialized: list = []
        self._track_terminated: list = []
        self._labels: list = [] # List of DO IDs and labels
        self._means: list = []
        self._covs: list = []
        self._length_upd: list = []  # List of DO length estimates. Assumed known
        self._width_upd: list = []  # List of DO width estimates. Assumed known
        self._NIS: list = []

        
        self._manager: Manager = setup_manager(self._params.IMM_off, self._params.single_target, self._params.visibility_off)
        self._sorted_track_indexes: list = []

    def track(self, t: float, dt: float, true_do_states: list, ownship_state: np.ndarray) -> Tuple[list, list]:
        """Tracks/updates estimates on dynamic obstacles, based on sensor measurements
        generated from the input true dynamic obstacle states.

        Args:
            dt (float): Time since last update
            t (float): Current time (assumed >= 0)
            true_do_states (list): List of tuples of true dynamic obstacle indices and states (do_idx, [x, y, Vx, Vy], length, width) x n_do. Used for simulating sensor measurements.
            ownship_state (np.ndarray): Ownship state vector [x, y, Vx, Vy] used for simulating sensor measurements.

        Returns:
            Tuple[list, list]: List of updated dynamic obstacle tracks (ID, state, cov, length, width). Also, a list the sensor measurements used.
        """
        # Update tracker variables based on true states
        max_sensor_range = max([sensor.max_range for sensor in self.sensors]) #Find largest range among the sensors
        for do_idx, do_state, do_length, do_width in true_do_states: # Loop through every DO, and get their ID, state, lenght and width
            dist_ownship_to_do = np.linalg.norm(do_state[:2] - ownship_state[:2]) #Calculate the distance between ownship and DO
            if do_idx not in self._labels and dist_ownship_to_do < max_sensor_range: #Check if DO is not already detected and is closer than max sensor range
                # New track. TODO: Implement track initiation, e.g. n out of m based initiation.
                self._labels.append(do_idx) #Add DO-ID to tracker Labels
                self._track_initialized.append(False) #? Why are these false still?ownship
                self._track_terminated.append(False) #? Why is this false still?
                self._means.append(np.array([0,0,0,0]))
                self._covs.append(np.eye(5))
                self._length_upd.append(do_length) #Include the length of object into tracker
                self._width_upd.append(do_width) #Include the width of object into tracker
                self._NIS.append(np.nan) #Don't include NIS yet
            elif do_idx in self._labels:
                self._track_initialized[self._labels.index(do_idx)] = True #Set the target as initialized


        
        sensor_measurements = []
        for sensor in self.sensors:
            if isinstance(sensor, Radar):
                z = sensor.generate_measurements(t, true_do_states, ownship_state)
                # print("z = : ", z , "type: ", type(z))
                sensor_measurements.append(z)
                meas_covariance_NE = sensor._params.R_cartesian #TODO: Get combined cov if necessary! (Not necessary accordin to Audun G. Hem (this cov is excessive (not used further)))
        
                # print("clutter = : ", clutter , "type: ", type(clutter))
                # print(sensor._params.to_dict())
        # meas_covariance_NE[0][0] = 10 # To see that the covariance comes out correct
        
        # TODO: Set this value via 
        # print(meas_covariance_XY)
            
        # TODO: Add functionality for several sensors

        # print(sensor_measurements)
        # Apply changes to measurements here so that they will fit into the VIMMJIPDA
        # The measurements already have added noise
        """ The ownship position needs to be a construct.State object.
        This comes on the form Construct.State(Mean, Covariance, timestamp, ID)
        Here, mean is the position on the form: (E, V_E, N, V_N, 0). Where E = East, N = North, V = Velocity
        Covariance = np.Identity(4)
        Timestamp = t
        Id can be skipped

        """
        ownship_mean = np.asarray([ownship_state[1], ownship_state[3], ownship_state[0], ownship_state[2], 0])
        # print(ownship_mean)
        ownship_cov = np.identity(5)

        ownship_pos = State(ownship_mean, ownship_cov, t)

        print("Testing that VIMMJIPDA is running")
        """
        The measurement set needs to be a set containing construct.Measurement object
        This comes on the form Construct.Measurement(measurement, measurement_params['cart_cov'],  float(timestamp))
        Measurement is the position in xy-coordinates on the form (E,N)
        measurement_params['cart_cov'] is given in parameters
        timestamp is given
        """

        #TODO: Look into, might need if measurements
        sensor_measurement = set() #Look at import_data.py to see how to transform data
        # print(type(sensor_measurement))

        new_meas = False # Create a variable to see if we are on a timestep that matches with sensor measurement rates
        for sensor in sensor_measurements:
            for meas in sensor:
                # print(meas, type(meas), " Time: ", t)
                # for do_idx, do_state, do_length, do_width in true_do_states:
                #     print(do_state[0], do_state[1])
                if not np.isnan(meas[0]) and not np.isnan(meas[1]):
                    values = np.asarray([meas[1],meas[0]])
                    meas_covariance_XY = np.asarray([[meas_covariance_NE[1][1], 0], [0, meas_covariance_NE[0][0]]])

                    # TODO: Add Functionality to choose if Filter knows the measurement Cov or not
                    # sensor_measurement.add(Measurement(values, measurement_params['cart_cov'],  t)) # Choose this if Tracker should not know meas cov
                    sensor_measurement.add(Measurement(values, meas_covariance_XY,  t)) # Choose this if Tracker should know meas cov
        
        # Check if this is a timestep with measurements
        for sensor in self.sensors:
            # Loop through all DO
            if isinstance(sensor, Radar):
                for i, (_, xs, length, width) in enumerate(true_do_states):
                    if ((t - sensor._prev_meas_time) % (1 / sensor._params.measurement_rate) == 0):
                        new_meas = True
            
        # Run the VIMMJIPDA Tracker at the same rate as sensor measurement rates
        if new_meas:
            self._manager.step(sensor_measurement, float(t), ownship=ownship_pos)

        for track in self._manager.tracks:
            if track.index not in self._sorted_track_indexes:
                self._sorted_track_indexes.append(track.index)
                self._sorted_track_indexes.sort()
        
        
        tracks = []
        # for track in self._manager.tracks:
        #     print("Timestep: ", t , " ",  track)
        
        for track in self._manager.tracks:
            # if track.index > len(self._means):
            #     self._means.append(np.array([0,0,0,0]))
            #     print("Test1")
            # if track.index > len(self._covs):
            #     self._covs.append(np.eye(4))
            #     print("Test2")
            # if track.index > len(self._length_upd):
            #     self._length_upd.append(true_do_states[0][2])
            # if track.index > len(self._width_upd):
            #     self._width_upd.append(true_do_states[0][3])


            # print(type(track))
            mean_xy, cov_xy = track.states.get_mean_covariance_array()
            # print(track.states.__len__())
            # print(track.states.leaves.get_mean_covariance_array())
            #print('\n mean: \n', mean_xy, type(mean_xy))
            #print('\n cov: \n', cov_xy, type(cov_xy))
            # print(cov_xy)
            mean_NE = np.array([mean_xy[0][2], mean_xy[0][0], mean_xy[0][3], mean_xy[0][1]])
            # TODO: Transform cov matrix to NE coordinates 
            # TODO: Create func to transform from XY to NE
            cov_NE = np.array([
                        [cov_xy[0][2][2], cov_xy[0][0][2], cov_xy[0][2][3] , cov_xy[0][2][1]],
                        [cov_xy[0][0][2], cov_xy[0][0][0], cov_xy[0][0][3], cov_xy[0][0][1]],
                        [cov_xy[0][2][3], cov_xy[0][0][3], cov_xy[0][3][3], cov_xy[0][3][1]],
                        [cov_xy[0][1][2], cov_xy[0][0][1], cov_xy[0][3][1], cov_xy[0][1][1]]
                ])
            # print(track.index)
            #self._means[track.index - 1] = mean_NE
            #self._covs[track.index - 1] = cov_NE
            # print(mean_NE, type(mean_NE), 'mean \n')

            
            # print('cov xy\n',cov_xy, type(cov_xy), '\n')
            # print('cov NE\n',cov_NE, type(cov_NE), '\n')
        # print(true_do_states, 'true states')
        # print(self._labels, 'labels')
        # print(self._labels, 'labels')
        #TODO: Move this into loop for more tracks than 1
            
        # print(self._labels, 'labels')  
        #TODO: Move this into loop for more tracks than 1
            for i, index in enumerate(self._sorted_track_indexes):
                if track.index == index:

                    tracks.append(
                        (
                            i,
                            mean_NE,
                            cov_NE,
                            8,
                            3
                        )
                    )
        
        tracks.sort(key=lambda x: x[0])
        #Return tracks and sensor_measurements

        if t > 370:
            print("t: ", t)
            print("tracks: ", tracks)
            print("true_vel: ", true_do_states[1][1])
        return tracks, sensor_measurements




    def get_track_information(self) -> Tuple[list, list]:
        
        
        tracks = []

        for track in self._manager.tracks:
            
            mean_xy, cov_xy = track.states.get_mean_covariance_array()
            mean_NE = np.array([mean_xy[0][2], mean_xy[0][0], mean_xy[0][3], mean_xy[0][1]])
            cov_NE = np.array([
                        [cov_xy[0][2][2], cov_xy[0][0][2], cov_xy[0][2][3] , cov_xy[0][2][1]],
                        [cov_xy[0][0][2], cov_xy[0][0][0], cov_xy[0][0][3], cov_xy[0][0][1]],
                        [cov_xy[0][2][3], cov_xy[0][0][3], cov_xy[0][3][3], cov_xy[0][3][1]],
                        [cov_xy[0][1][2], cov_xy[0][0][1], cov_xy[0][3][1], cov_xy[0][1][1]]
                ])
            
            for i, index in enumerate(self._sorted_track_indexes):
                if track.index == index:

                    tracks.append(
                        (
                            i,
                            mean_NE,
                            cov_NE,
                            8,
                            3
                        )
                    )
        tracks.sort(key=lambda x: x[0])
        
        return tracks, self._NIS
