from colav_simulator.core.tracking.VIMMJIPDA_interface.VIMMJIPDA_testing.code.tracking import constructs, utilities, filters, models, initiators, terminators, managers, associators, trackers
from colav_simulator.core.tracking.VIMMJIPDA_interface.VIMMJIPDA_testing.code.parameters import tracker_params, measurement_params, process_params

from  colav_simulator.core.tracking.VIMMJIPDA_interface.VIMMJIPDA_testing.code import import_data
from  colav_simulator.core.tracking.VIMMJIPDA_interface.VIMMJIPDA_testing.code import plotting
import numpy as np


def setup_manager(IMM_off, single_target, visibility_off):
    if IMM_off:
        kinematic_models = [models.CVModel(process_params['cov_CV_high'])]
        pi_matrix = np.array([[1]])
        init_mode_probs = np.array([1])
    else:
        kinematic_models = [models.CVModel(process_params['cov_CV_low']),models.CTModel(process_params['cov_CV_low'],process_params['cov_CT']),models.CVModel(process_params['cov_CV_high'])]
        pi_matrix = process_params['pi_matrix']
        init_mode_probs = process_params['init_mode_probs']

    clutter_model = models.ConstantClutterModel(tracker_params['clutter_density'])

    measurement_model = models.CombinedMeasurementModel(
        measurement_mapping = measurement_params['measurement_mapping'],
        cartesian_covariance = measurement_params['cart_cov'],
        range_covariance = measurement_params['range_cov'],
        bearing_covariance = measurement_params['bearing_cov'])

    filter = filters.IMMFilter(
        measurement_model = measurement_model,
        mode_transition_matrix = pi_matrix)

    data_associator = associators.MurtyDataAssociator(
        n_measurements_murty = 4,
        n_tracks_murty = 2,
        n_hypotheses_murty = 8)

    tracker = trackers.VIMMJIPDATracker(
        filter,
        clutter_model,
        data_associator,
        survival_probability=tracker_params['survival_prob'],
        visibility_transition_matrix = tracker_params['visibility_transition_matrix'],
        detection_probability=tracker_params['P_D'],
        gamma=tracker_params['gamma'],
        single_target=single_target,
        visibility_off=visibility_off)

    track_initiation = initiators.SinglePointInitiator(
        tracker_params['init_prob'],
        measurement_model,
        tracker_params['init_Pvel'],
        mode_probabilities = init_mode_probs,
        kinematic_models = kinematic_models,
        visibility_probability = 0.9)

    track_terminator = terminators.Terminator(
        tracker_params['term_threshold'],
        max_steps_without_measurements = 5,
        fusion_significance_level = 0.01)

    track_manager = managers.Manager(tracker, track_initiation, track_terminator, tracker_params['conf_threshold'])
    return track_manager



if __name__ == '__main__':
    """
    All tracker parameters are imported from parameters.py, and can be changed
    there.
    """
    # choose data set
    joyride = False
    final_dem = not joyride

    # select the part of the data sets to import
    if joyride:
        t_max = 10000
        t_min = 0

    if final_dem:
        t_max = 1300
        t_min = 900

    # turn off tracker functionality
    IMM_off = True
    single_target = True
    visibility_off = True

    # import data
    if joyride:
        measurements, ownship, ground_truth, timestamps = import_data.joyride(t_min=t_min, t_max=t_max)

    if final_dem:
        measurements, ownship, ground_truth, timestamps = import_data.final_dem(t_min=t_min, t_max=t_max) #ground_truth here refers to the Gunnerus AIS data

    # define tracker evironment
    manager = setup_manager(IMM_off, single_target, visibility_off)


    # run tracker
    for k, (measurement_set, timestamp, ownship_pos) in enumerate(zip(measurements, timestamps, *ownship.values())):
        # print(f'Timestep {k}:')
        # print(measurement_set, type(measurement_set))
        # print(measurements, type(measurements))
        # print(ownship_pos)
        # print(timestamp)
        #for meas in measurement_set:
        #    print(meas, type(meas))
        manager.step(measurement_set, float(timestamp), ownship=ownship_pos)
        # print(f'Active tracks: {np.sort([track.index for track in manager.tracks])}\n')




    # plotting
    plot = plotting.ScenarioPlot(
        measurement_marker_size=3,
        track_marker_size=5,
        add_covariance_ellipses=True,
        add_validation_gates=False,
        add_track_indexes=False,
        gamma=3.5
    )
    plot.create(measurements, manager.track_history, ownship, timestamps, ground_truth)



