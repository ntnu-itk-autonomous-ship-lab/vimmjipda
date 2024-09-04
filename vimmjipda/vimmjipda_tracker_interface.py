"""
    File: vimmjipda_tracker_interface.py

    Summary:
        This file contains a colav-simulator tracker interface to the VIMMJIPDA tracker made by Audun Gulliksstad Hem, Edmund Førland Brekke and Lars-Christian Ness Tokle.

    Author: Ragnar Wien, Trym Tengesdal
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import colav_simulator.common.math_functions as mf
import colav_simulator.core.sensing as cs_sensing
import colav_simulator.core.tracking.trackers as cs_trackers
import numpy as np
import vimmjipda.code.tracking.constructs as constructs
import vimmjipda.code.tracking.managers as managers
import yaml
from vimmjipda.code.setup import setup_manager


@dataclass
class VIMMJIPDAParams:
    """Class for holding VIMMJIPDA parameters."""

    enable_imm: bool = field(default_factory=lambda: False)
    single_target: bool = field(default_factory=lambda: False)
    enable_visibility: bool = field(default_factory=lambda: False)

    def to_dict(self):
        output_dict = {
            "enable_imm": self.enable_imm,
            "single_target": self.single_target,
            "enable_visibility": self.enable_visibility,
        }
        return output_dict

    @classmethod
    def from_dict(cls, config_dict):
        return VIMMJIPDAParams(
            enable_imm=config_dict["enable_imm"],
            single_target=config_dict["single_target"],
            enable_visibility=config_dict["enable_visibility"],
        )

    @classmethod
    def from_yaml(cls, file: Path):
        with open(file, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


class VIMMJIPDA(cs_trackers.ITracker):
    """The VIMMJIPDA class implements the VIMMJIPDA (Visibility Interacting Multiple Models Joint Integrated Probabilistic Data Association) tracker by
    Audun Gulliksstad Hem, Edmund Førland Brekke and Lars-Christian Ness Tokle, introduced in the article:
    "Multitarget Tracking With Multiple Models and Visibility: Derivation and Verification on Maritime Radar Data"

    NOTE: It is possible to configure this tracker by turning of the functionality for: Interacting Multiple Models, Visibility and Multi-Target turning
    the tracker into an IPDA tracker

    NOTE: Only supports the Radar sensor at the moment, and needs to be checked for bugs.
    """

    def __init__(
        self, sensor_list: Optional[List[cs_sensing.ISensor]] = None, params: Optional[VIMMJIPDAParams] = None
    ) -> None:
        if params is not None:
            self._params: VIMMJIPDAParams = params
        else:
            self._params = VIMMJIPDAParams()
        self.sensors: list = sensor_list

        self._en_to_ne_pmatrix: np.ndarray = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]])
        self._t_prev: float = 0.0
        self._recent_sensor_measurements: list = []

        self._manager: managers.Manager = setup_manager(
            IMM_off=not self._params.enable_imm,
            single_target=self._params.single_target,
            visibility_off=not self._params.enable_visibility,
        )
        self._sorted_track_indexes: list = []

    def set_sensor_list(self, sensor_list: List[cs_sensing.ISensor]) -> None:
        self.sensors = sensor_list

    def track(
        self, t: float, dt: float, true_do_states: List[Tuple[int, np.ndarray, float, float]], ownship_state: np.ndarray
    ) -> Tuple[List[Tuple[int, np.ndarray, np.ndarray, float, float]], List[Tuple[int, np.ndarray]]]:
        assert self.sensors is not None, "Sensor list is not set."
        if t <= self._t_prev:
            tracks, _ = self.get_track_information(ownship_state)
            return tracks, self._recent_sensor_measurements

        self._t_prev = t
        sensor_measurements = []
        for sensor in self.sensors:
            if isinstance(sensor, cs_sensing.Radar):
                z = sensor.generate_measurements(t, true_do_states, ownship_state)
                sensor_measurements.append(z)
                meas_covariance_NE = sensor.params.R_ne
        self._recent_sensor_measurements = sensor_measurements

        # The ownship position needs to be a construct.State object.
        # This comes on the form Construct.State(Mean, Covariance, timestamp, ID)
        # Here, mean is the position on the form: (E, V_E, N, V_N, 0). Where E = East, N = North, V = Velocity
        # Covariance = np.Identity(4)
        # Timestamp = t and ID can be skipped
        os_course = np.arctan2(ownship_state[3], ownship_state[2])
        os_speed = np.linalg.norm(ownship_state[2:4])
        ownship_mean = np.asarray(
            [ownship_state[1], os_speed * np.sin(os_course), ownship_state[0], os_speed * np.cos(os_course), 0]
        )
        ownship_cov = np.identity(5)
        ownship_pos = constructs.State(ownship_mean, ownship_cov, t)

        # The measurement set needs to be a set containing construct.Measurement object
        # This comes on the form Construct.Measurement(measurement, measurement_params['cart_cov'],  float(timestamp))
        # Measurement is the position in xy-coordinates on the form (E,N)
        # measurement_params['cart_cov'] is given in parameters
        # timestamp is given
        sensor_measurement = set()
        new_meas = False
        meas_covariance_EN = np.asarray([[meas_covariance_NE[1][1], 0], [0, meas_covariance_NE[0][0]]])
        for sensor in sensor_measurements:
            for meas_tup in sensor:
                if np.any(np.isnan(meas_tup[1])):
                    continue
                meas_EN = np.asarray([meas_tup[1][1], meas_tup[1][0]])
                sensor_measurement.add(constructs.Measurement(meas_EN, meas_covariance_EN, t))
                new_meas = True

        if new_meas:
            self._manager.step(sensor_measurement, float(t), ownship=ownship_pos)

        tracks = []
        for track in self._manager.tracks:
            mean_EN, cov_EN = track.states.get_mean_covariance_array()
            mean_NE = self._en_to_ne_pmatrix @ mean_EN[0][:4]
            cov_NE = self._en_to_ne_pmatrix @ cov_EN[0][:4, :4] @ self._en_to_ne_pmatrix.T
            if not self.check_estimate_extremity(mean_NE, cov_NE):
                tracks.append(
                    (track.index, mean_NE, cov_NE, 10.0, 3.0)
                )  # default values for length and width as they are not estimated

        tracks_sorted_by_distance = sorted(tracks, key=lambda x: np.linalg.norm(x[1][:2] - ownship_state[:2]))
        return tracks_sorted_by_distance, sensor_measurements

    def check_estimate_extremity(self, mean: np.ndarray, cov: np.ndarray) -> bool:
        """Check for extreme velocity estimates and unreasonably high covariances

        Args:
            mean (np.ndarray): The mean of the state estimate
            cov (np.ndarray): The covariance of the state estimate

        Returns:
            bool: True if the estimate is unreasonable, False otherwise
        """
        if (
            abs(mean[2]) > 12.0
            or abs(mean[3]) > 12.0
            or cov[0, 0] > 50.0**2
            or cov[1, 1] > 50.0**2
            or cov[2, 2] > 4.0
            or cov[3, 3] > 4.0
        ):
            return True
        return False

    def get_track_information(
        self, ownship_state: np.ndarray
    ) -> Tuple[List[Tuple[int, np.ndarray, np.ndarray, float, float]], List[float]]:
        tracks = []
        for track in self._manager.tracks:
            mean_EN, cov_EN = track.states.get_mean_covariance_array()
            mean_NE = self._en_to_ne_pmatrix @ mean_EN[0][:4]
            cov_NE = self._en_to_ne_pmatrix @ cov_EN[0][:4, :4] @ self._en_to_ne_pmatrix.T

            tracks.append((track.index, mean_NE, cov_NE, 10.0, 3.0))

        tracks_sorted_by_distance = sorted(tracks, key=lambda x: np.linalg.norm(x[1][:2] - ownship_state[:2]))
        return tracks_sorted_by_distance, np.nan * np.zeros(len(tracks_sorted_by_distance))

    def reset(self) -> None:
        self._manager.reset()
        self._t_prev = 0.0
