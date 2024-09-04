import numpy as np
import vimmjipda.code.parameters as vimmjipda_parameters
import vimmjipda.code.tracking.associators as vimmjipda_associators
import vimmjipda.code.tracking.filters as vimmjipda_filters
import vimmjipda.code.tracking.initiators as vimmjipda_initiators
import vimmjipda.code.tracking.managers as vimmjipda_managers
import vimmjipda.code.tracking.models as vimmjipda_models
import vimmjipda.code.tracking.terminators as vimmjipda_terminators
import vimmjipda.code.tracking.trackers as vimmjipda_trackers


def setup_manager(IMM_off: bool, single_target: bool, visibility_off: bool) -> vimmjipda_managers.Manager:
    if IMM_off:
        kinematic_models = [vimmjipda_models.CVModel(vimmjipda_parameters.process_params["cov_CV_high"])]
        pi_matrix = np.array([[1]])
        init_mode_probs = np.array([1])
    else:
        kinematic_models = [
            vimmjipda_models.CVModel(vimmjipda_parameters.process_params["cov_CV_low"]),
            vimmjipda_models.CTModel(
                vimmjipda_parameters.process_params["cov_CV_low"], vimmjipda_parameters.process_params["cov_CT"]
            ),
            vimmjipda_models.CVModel(vimmjipda_parameters.process_params["cov_CV_high"]),
        ]
        pi_matrix = vimmjipda_parameters.process_params["pi_matrix"]
        init_mode_probs = vimmjipda_parameters.process_params["init_mode_probs"]

    clutter_model = vimmjipda_models.ConstantClutterModel(vimmjipda_parameters.tracker_params["clutter_density"])

    measurement_model = vimmjipda_models.CombinedMeasurementModel(
        measurement_mapping=vimmjipda_parameters.measurement_params["measurement_mapping"],
        cartesian_covariance=vimmjipda_parameters.measurement_params["cart_cov"],
        range_covariance=vimmjipda_parameters.measurement_params["range_cov"],
        bearing_covariance=vimmjipda_parameters.measurement_params["bearing_cov"],
    )

    tfilter = vimmjipda_filters.IMMFilter(measurement_model=measurement_model, mode_transition_matrix=pi_matrix)

    data_associator = vimmjipda_associators.MurtyDataAssociator(
        n_measurements_murty=4, n_tracks_murty=2, n_hypotheses_murty=8
    )

    tracker = vimmjipda_trackers.VIMMJIPDATracker(
        tfilter,
        clutter_model,
        data_associator,
        survival_probability=vimmjipda_parameters.tracker_params["survival_prob"],
        visibility_transition_matrix=vimmjipda_parameters.tracker_params["visibility_transition_matrix"],
        detection_probability=vimmjipda_parameters.tracker_params["P_D"],
        gamma=vimmjipda_parameters.tracker_params["gamma"],
        single_target=single_target,
        visibility_off=visibility_off,
    )

    track_initiation = vimmjipda_initiators.SinglePointInitiator(
        vimmjipda_parameters.tracker_params["init_prob"],
        measurement_model,
        vimmjipda_parameters.tracker_params["init_Pvel"],
        mode_probabilities=init_mode_probs,
        kinematic_models=kinematic_models,
        visibility_probability=0.9,
    )

    track_terminator = vimmjipda_terminators.Terminator(
        vimmjipda_parameters.tracker_params["term_threshold"],
        max_steps_without_measurements=5,
        fusion_significance_level=0.01,
    )

    track_manager = vimmjipda_managers.Manager(
        tracker, track_initiation, track_terminator, vimmjipda_parameters.tracker_params["conf_threshold"]
    )
    return track_manager
