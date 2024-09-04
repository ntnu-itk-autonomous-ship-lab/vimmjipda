import vimmjipda.code.setup as vimmjipda_setup

if __name__ == "__main__":
    from vimmjipda.code import import_data, plotting

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
        measurements, ownship, ground_truth, timestamps = import_data.final_dem(
            t_min=t_min, t_max=t_max
        )  # ground_truth here refers to the Gunnerus AIS data

    # define tracker evironment
    manager = vimmjipda_setup.setup_manager(IMM_off, single_target, visibility_off)

    # run tracker
    for k, (measurement_set, timestamp, ownship_pos) in enumerate(zip(measurements, timestamps, *ownship.values())):
        # print(f'Timestep {k}:')
        # print(measurement_set, type(measurement_set))
        # print(measurements, type(measurements))
        # print(ownship_pos)
        # print(timestamp)
        # for meas in measurement_set:
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
        gamma=3.5,
    )
    plot.create(measurements, manager.track_history, ownship, timestamps, ground_truth)
