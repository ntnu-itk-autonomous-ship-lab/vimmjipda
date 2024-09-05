"""
    Test module for the VIMMJIPDA interface class. NOTE: The colav-simulator must be installed to run this test.
"""

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.core.sensing as sensing
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sgeo
import vimmjipda.paths as paths
import vimmjipda.vimmjipda_tracker_interface as vti


def test_vimmjipda_interface() -> None:
    rparams = sensing.RadarParams()
    rparams.generate_clutter = True
    rparams.detection_probability = 0.95
    rparams.measurement_rate = 0.5
    rparams.max_range = 1000.0
    radar = sensing.Radar(rparams)
    radar.reset(seed=0)

    vimmjipda_params = vti.VIMMJIPDAParams.from_yaml(paths.config / "vimmjipda.yaml")
    vimmjipda_tracker = vti.VIMMJIPDA([radar], vimmjipda_params)

    ownship_state = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    true_do_states = [
        (0, np.array([50.0, 0.0, 0.0, 2.0]), 10.0, 3.0)
    ]  # dynamic obstacle info on the form (ID, state, length, width)

    matplotlib.use("TkAgg")
    fig, ax = plt.subplots()
    plt.ion()
    plt.show(block=False)
    ax.set_aspect("equal")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_xlim(-50.0, 0.2 * rparams.max_range)
    ax.set_ylim(0.0, 0.2 * rparams.max_range)

    dt = 0.5
    track_data = {}
    track_labels = []
    ell_handles = []
    os_handle = None
    do_handle = None
    meas_handles = []
    state_handles = []
    for k in range(100):
        t = k * dt

        ownship_state_vxvy = mhm.convert_state_to_vxvy_state(ownship_state)
        tracks, meas = vimmjipda_tracker.track(t, dt, true_do_states, ownship_state_vxvy)

        for track in tracks:
            if track[0] not in track_labels:
                track_labels.append(track[0])
                track_data[f"Track{track[0]}"] = []
                track_data[f"Track{track[0]}"].append((track[1], track[2]))
                print(f"New track with ID {track[0]} at time {t}")
            else:
                idx = track_labels.index(track[0])
                track_data[f"Track{track[0]}"].append((track[1], track[2]))

        # Clear plot handles before updating
        for prev_state_handle, prev_ell_handle in zip(state_handles, ell_handles):
            try:
                prev_state_handle.remove()
                prev_ell_handle.remove()
            except:
                continue
        if os_handle is not None:
            os_handle.remove()
            do_handle.remove()

        # Then plot new measurements, tracks and ships
        for sensor in meas:
            for sensor_meas_tup in sensor:
                if np.any(np.isnan(sensor_meas_tup[1])):
                    continue
                mcolor = "co"
                if sensor_meas_tup[0] >= 0:
                    mcolor = "ro"
                hdl = ax.plot(sensor_meas_tup[1][1], sensor_meas_tup[1][0], mcolor, markersize=5)
                meas_handles.append(hdl[0])

        for td in track_data.values():
            states = np.array([tup[0] for tup in td]).T
            last_cov = td[-1][1]
            ellipse_x, ellipse_y = mhm.create_probability_ellipse(last_cov, probability=0.65)
            ell_geometry = sgeo.Polygon(zip(ellipse_y + states[1, -1], ellipse_x + states[0, -1]))
            ell_handles.append(ax.fill(*ell_geometry.exterior.xy, color="green", alpha=0.3)[0])

            state_handles.append(ax.plot(states[1, :], states[0, :])[0])

        os_poly = mapf.create_ship_polygon(
            x=ownship_state[0], y=ownship_state[1], heading=ownship_state[2], length=10.0, width=2.0
        )
        os_handle = ax.fill(*os_poly.exterior.xy, color="b", alpha=0.5)[0]

        do_poly = mapf.create_ship_polygon(
            x=true_do_states[0][1][0], y=true_do_states[0][1][1], heading=np.pi / 2, length=10.0, width=3.0
        )
        do_handle = ax.fill(*do_poly.exterior.xy, true_do_states[0][1][0], "r")[0]

        # Lastly, perform simple Euler integration
        ownship_state = ownship_state + dt * np.array(
            [
                ownship_state[3] * np.cos(ownship_state[2]),
                ownship_state[3] * np.sin(ownship_state[2]),
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        do_state = true_do_states[0][1] + dt * np.array(
            [
                true_do_states[0][1][2],
                true_do_states[0][1][3],
                0.0,
                0.0,
            ]
        )
        true_do_states = [(0, do_state, 10.0, 3.0)]

        plt.pause(0.1)

    print("Num tracks after the run: ", len(vimmjipda_tracker.get_track_information(ownship_state)[0]))

    vimmjipda_tracker.reset()


if __name__ == "__main__":
    test_vimmjipda_interface()
