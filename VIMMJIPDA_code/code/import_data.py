import scipy.io as io
import numpy as np
from colav_simulator.core.tracking.VIMMJIPDA_interface.VIMMJIPDA_testing.code.tracking import constructs
from colav_simulator.core.tracking.VIMMJIPDA_interface.VIMMJIPDA_testing.code.parameters import measurement_params


def NE_to_xy(state_list):
    """
    Convert from North-East coordinates to xy-coordinates.
    """
    if len(state_list[0]) == 2:
        transform = np.array([[0,1],[1,0]])
    elif len(state_list[0]) == 4:
        transform = np.array([[0,0,1,0],
                              [0,0,0,1],
                              [1,0,0,0],
                              [0,1,0,0]])
    elif len(state_list[0]) == 5:
        transform = np.array([[0,0,1,0,0],
                              [0,0,0,1,0],
                              [1,0,0,0,0],
                              [0,1,0,0,0],
                              [0,0,0,0,0]])
    else:
        print('Wrong data input shape.')
        exit()

    return np.matmul(state_list, transform)


def ensure_correct_state_dimension(state_list):
    pad_amount = len(measurement_params['measurement_mapping'][0])-len(state_list[0])
    return np.pad(state_list, [(0, pad_amount), (0, pad_amount)])


def final_dem(t_min=0, t_max=10000):
    mat = io.loadmat('/home/ragnarnw/Github/colav_simulator/colav_simulator/core/tracking/VIMMJIPDA/data/final_demo.mat') #Using absolute path
    for key, value in mat.items():
        if key == 'measurements':
            measurements = np.asarray(value)[0]
            for i, measurement_set in enumerate(measurements):
                delete_indices = []
                for j, measurement in enumerate(measurement_set):
                    if np.any(np.isinf(measurement)):
                        delete_indices.append(j)
                #print('\n\n before del \n')
                #print(measurement_set)
                measurement_set = np.delete(measurement_set, delete_indices, axis=-2)
                #print('\n\n after del \n')
                #print(measurement_set)
                measurements[i] = NE_to_xy(measurement_set)
                #print('\n\n in array: \n')
                #print(measurements[i])
        if key == 'timestamps':
            timestamps = np.asarray(value)[0]
        if key == 'TELEMETRON':
            ownship = np.asarray(value)
            # print(ownship[0])
            ownship = ensure_correct_state_dimension(ownship)
            #print(ownship)
            ownship = NE_to_xy(ownship)
            # print(ownship[0])
            #for ownship_pos in ownship:
            #    print(ownship_pos)
        if key == 'GUNNERUS':
            gunnerus_ais = np.asarray(value)
            gunnerus_ais = ensure_correct_state_dimension(gunnerus_ais)
            gunnerus_ais = NE_to_xy(gunnerus_ais)
    timestamps = timestamps-timestamps[0]
    valid_indexes = np.where((t_min <= timestamps.squeeze()) & (timestamps.squeeze() <= t_max))
    timestamps = timestamps[valid_indexes]
    print(len(timestamps))
    measurements_all = np.array([set() for i in valid_indexes[0]])
    print(measurements_all)

    for i, (measurement_set, timestamp) in enumerate(zip(measurements[valid_indexes], timestamps)):
        print(measurement_set)
        for measurement in measurement_set:
            #print(measurement)
            measurements_all[i].add(constructs.Measurement(measurement, measurement_params['cart_cov'],  float(timestamp)))

    gunnerus_ais = {1: [constructs.State(measurement, np.identity(4), timestamp) for measurement, timestamp in zip(gunnerus_ais[valid_indexes], timestamps)]}
    ownship = {1: [constructs.State(ownship_pos, np.identity(4), timestamp) for ownship_pos, timestamp in zip(ownship[valid_indexes], timestamps)]}
    return measurements_all, ownship, gunnerus_ais,timestamps

def joyride(t_min=0, t_max=10000):
    data = io.loadmat('/home/ragnarnw/Github/colav_simulator/colav_simulator/core/tracking/VIMMJIPDA/data/joyride.mat')

    for key, value in data.items():
        if key == 'measurements':
            measurements = np.asarray(value).squeeze()
            for i, measurement_set in enumerate(measurements):
                delete_indices = []
                for j, measurement in enumerate(measurement_set):
                    if np.any(np.isinf(measurement)):
                        delete_indices.append(j)
                measurement_set = np.delete(measurement_set, delete_indices, axis=-2)
                measurements[i] = NE_to_xy(measurement_set)
        if key == 'target':
            gt = np.asarray(value)
            gt = ensure_correct_state_dimension(gt)
            gt = NE_to_xy(gt)
        if key == 'telemetron':
            ownship = np.asarray(value)
            ownship = ensure_correct_state_dimension(ownship)
            ownship = NE_to_xy(ownship)
        if key == 'time':
            timestamps = np.asarray(value)

    timestamps = timestamps-timestamps[0]
    valid_indexes = np.where((t_min <= timestamps.squeeze()) & (timestamps.squeeze() <= t_max))
    timestamps = timestamps[valid_indexes]
    measurements_all = np.array([set() for i in valid_indexes[0]])

    for i, (measurement_set, timestamp) in enumerate(zip(measurements[valid_indexes], timestamps)):
        for measurement in measurement_set:
            measurements_all[i].add(constructs.Measurement(measurement, measurement_params['cart_cov'],  float(timestamp)))
                
    ground_truth = {1: [constructs.State(gt, np.identity(4), timestamp) for gt, timestamp in zip(gt[valid_indexes], timestamps)]}
    telemetron = {1: [constructs.State(ownship_pos, np.identity(4), timestamp) for ownship_pos, timestamp in zip(ownship[valid_indexes], timestamps)]}
    return measurements_all, telemetron, ground_truth, timestamps
