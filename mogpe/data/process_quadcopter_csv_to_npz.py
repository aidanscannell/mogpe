import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# rotation and translation of start positions
angles = [np.pi / 2, np.pi, 0, 0]


def vicon_csv_to_npz_data(step,
                          num_trim=10,
                          path_to_csv_folder=None,
                          plot_each_csv=False):
    """Process folder of csv files and return inputs and outputs.

    :param step: Vicon data is collected at 100Hz. Step defines how many data
            points to ignore between processing.
    :param num_trim: number of data points to trim from start and end.
    :param path_to_csv_folder: each csv file should represent a single trial.
    :param save_npz_filename: where to save the numpy file.
    :param plot_each_csv: if True, plots each csv trial.

    :returns: inputs [num_data, 2] The inputs are x and y coordinates.
              outputs [num_data, 3] The outputs as delta x, delta y, delta z
    """
    filename_list = glob.glob(path_to_csv_folder + '/*vicon-data*.csv')
    inputs, outputs = [], []
    for filename in filename_list:
        input, output = process_single_csv(step,
                                           filename,
                                           plot_flag=plot_each_csv)
        inputs.append(input[num_trim:-num_trim])
        outputs.append(output[num_trim:-num_trim])
    inputs = np.concatenate(inputs)
    outputs = np.concatenate(outputs)

    # Remove data where tello is flying along y axis
    # x2_low = -3.
    x2_low = -2.5
    x2_high = 2.
    x1_high = 3.
    mask_1 = inputs[:, 1] > x2_low
    mask_3 = inputs[:, 1] < x2_high
    mask_2 = inputs[:, 0] < x1_high
    mask = mask_1 & mask_3 & mask_2
    inputs = inputs[mask, :]
    outputs = outputs[mask, :]

    print('Input data shape: %s' % str(inputs.shape))
    print('Output data shape: %s' % str(outputs.shape))
    return inputs, outputs


def process_single_csv(step, filename, plot_flag):
    print('Parsing: %s' % filename)
    data = pd.read_csv(filename, index_col=0)
    # data_np = data.to_numpy()
    data_np = data.to_numpy()
    num_data = data_np.shape[0]

    # Create dictionary for data
    data_dict = {}
    for idx, col in enumerate(data.columns):
        data_dict[col] = data_np[:, idx]

    # Find the index to start trial from
    start_idx = 0
    for i in range(num_data):
        if data_dict['vicon_x'][i] < -1.8 and data_dict['vicon_y'][i] > -1:
            start_idx = i
            break

    # Recreate dictionary with only data from start of trial
    data_dict = {}
    for idx, col in enumerate(data.columns):
        data_dict[col] = data_np[start_idx:, idx]

    # Determine which corner of the flight lab it started from
    if data_dict['vicon_x'][0] > 0 and data_dict['vicon_y'][0] > 0:
        start_pos = 4
    elif data_dict['vicon_x'][0] < 0 and data_dict['vicon_y'][0] > 0:
        start_pos = 2
    elif data_dict['vicon_x'][0] > 0 and data_dict['vicon_y'][0] < 0:
        start_pos = 3
    elif data_dict['vicon_x'][0] < 0 and data_dict['vicon_y'][0] < 0:
        start_pos = 1
    print('start position: %i' % start_pos)

    # Rotate and translate the data so that all trials line up
    tello_x, tello_y, data_dict = rotate_and_translate(start_pos, data_dict)
    data_dict['tello_x'] = tello_x
    data_dict['tello_y'] = tello_y

    dx, dy, dz, n_steps_array, tello_zero_idx, dxyz, drz = calc_error(
        data_dict, step, tello_x, tello_y)

    if data_dict['vicon_y'][0::step].shape != dy.shape:
        diff_x = data_dict['vicon_x'][0::step].shape[0] - dx.shape[0]
        diff_y = data_dict['vicon_y'][0::step].shape[0] - dy.shape[0]
        diff_z = data_dict['vicon_z'][0::step].shape[0] - dz.shape[0]
        dx = dx[:diff_x]
        dy = dy[:diff_y]
        dz = dz[:diff_z]
        dxyz = dxyz[:diff_z]
        drz = drz[:diff_z]

    if start_pos == 1:
        dxyz = dx
    elif start_pos == 2:
        dxyz = dx
    elif start_pos == 3:
        dxyz = dx
    elif start_pos == 4:
        dxyz = dy

    inputs = np.stack(
        [data_dict['vicon_x'][0::step], data_dict['vicon_y'][0::step]]).T
    # outputs = np.stack([dx, dy, dz, dxyz]).T
    outputs = np.stack([dx, dy, dz]).T

    x = data_dict['vicon_x'][0::step]
    y = data_dict['vicon_y'][0::step]

    if plot_flag:
        fig = plt.figure(figsize=(12, 12))
        plt.quiver(
            x,
            y,
            dx,
            # dxyz,
            np.zeros([*dxyz.shape]),
            angles='xy',
            scale_units='xy',
            width=0.001,
            scale=1,
            zorder=10)
        plt.plot(data_dict['vicon_x'],
                 data_dict['vicon_y'],
                 label='vicon',
                 color="darkmagenta")
        plt.axis('off')
        save_name = '../features/data/quadcopter/trajectory.png'
        plt.savefig(save_name, transparent=True, bbox_inches='tight')
        plt.show(block=True)
    return inputs, outputs


def rotate_and_translate(start_pos, data_dict):
    tello_pos = np.stack(
        [data_dict['tello_x'], data_dict['tello_y'], data_dict['tello_z']],
        axis=1)
    theta = angles[start_pos - 1]
    R = np.array([[np.cos(theta), np.sin(theta), 0],
                  [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    rotated_tello_xyz = tello_pos @ R
    rotated_tello_xyz[:, 0] *= -1
    rotated_tello_xyz[:, 0] += data_dict['vicon_x'][0]
    rotated_tello_xyz[:, 1] += data_dict['vicon_y'][0]
    rotated_tello_xyz[:, 2] += data_dict['vicon_z'][0]
    data_dict['vicon_rz'] -= data_dict['vicon_rz'][0]
    # data_dict['vicon_rz'] = -data_dict['vicon_rz']

    return rotated_tello_xyz[:, 0], rotated_tello_xyz[:, 1], data_dict


def calc_error(data_dict, step, tello_x, tello_y):
    N = round(tello_x.shape[0] / step)
    dx = np.zeros(N + 1)
    dy = np.zeros(N + 1)
    dz = np.zeros(N + 1)
    dxyz = np.zeros(N + 1)
    drz = np.zeros(N + 1)
    n_steps_array = np.zeros(N + 1)
    # dx_sum = 0
    # dy_sum = 0
    # dz_sum = 0
    counter = 0
    tello_zero_idx = []
    n_steps = 1
    for i in range(1, tello_x.shape[0] - 1)[::step]:
        diff_rz = data_dict['vicon_rz'][i] - data_dict['tello_rz'][i]
        drz[counter] = abs(diff_rz)
        if int(tello_x[i] * 1) == int(data_dict['vicon_x'][0] * 1) and int(
                tello_y[i] * 1) == int(data_dict['vicon_y'][0] * 1):
            # if int(tello_x[i] * 100) == int(data_dict['vicon_x'][0] * 100) or int(
            #         tello_x[i - step] * 100) == int(data_dict['vicon_x'][0] * 100):
            # print('ZERO')
            # print('i: %i' % i)
            tello_zero_idx.append(i)
            n_steps += 1
            # print(n_steps)
        else:
            # print('NON ZERO')
            dx[counter] = (data_dict['vicon_x'][i] -
                           data_dict['vicon_x'][i - step * n_steps])
            #- (tello_x[i] - tello_x[i - step * n_steps])) / n_steps
            dy[counter] = (data_dict['vicon_y'][i] -
                           data_dict['vicon_y'][i - step * n_steps])
            #- (tello_y[i] - tello_y[i - step * n_steps])) / n_steps
            dz[counter] = (data_dict['vicon_z'][i] -
                           data_dict['vicon_z'][i - step * n_steps])

            n_steps = 1
        n_steps_array[counter] = int(n_steps)
        counter += 1

    return dx, dy, dz, n_steps_array, tello_zero_idx, dxyz, drz


if __name__ == "__main__":
    path_to_csv_folder = '../../data/raw/quadcopter/27-feb/half-lengthscale'
    save_npz_filename = '../../data/processed/quadcopter_turbulence_new.npz'
    step = 20
    # step = 40
    # step = 50
    plot_each_csv = False
    # plot_each_csv = True

    X, Y = vicon_csv_to_npz_data(step,
                                 path_to_csv_folder=path_to_csv_folder,
                                 plot_each_csv=plot_each_csv)

    # Save numpy file
    np.savez(save_npz_filename, x=X, y=Y)

    data = np.load(save_npz_filename)
    X = data['x']
    Y = data['y']

    fig = plt.figure(figsize=(12, 12))
    plt.quiver(X[:, 0],
               X[:, 1],
               Y[:, 0],
               np.zeros([*Y[:, 0].shape]),
               angles='xy',
               scale_units='xy',
               width=0.001,
               scale=1,
               zorder=10)
    plt.axis('off')
    save_name = '../features/data/quadcopter/quiver.png'
    plt.savefig(save_name, transparent=True, bbox_inches='tight')
    plt.show()
