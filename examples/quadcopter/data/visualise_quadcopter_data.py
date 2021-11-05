#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

params = {
    # 'axes.labelsize': 30,
    # 'font.size': 30,
    # 'legend.fontsize': 20,
    # 'xtick.labelsize': 30,
    # 'ytick.labelsize': 30,
    "text.usetex": True,
}
plt.rcParams.update(params)

load_npz_filename = "./quadcopter_data.npz"
load_npz_filename = "./quadcopter_data_step_20.npz"
# load_npz_filename = "./quadcopter_data_step_40.npz"
load_npz_filename = "./quadcopter_data_step_20_single_direction.npz"
load_npz_filename = "./quadcopter_data_step_40_direction_up.npz"
# load_npz_filename = "./quadcopter_data_step_40_direction_down.npz"

data = np.load(load_npz_filename)
X = data["x"]
Y = data["y"]

fig = plt.figure()
plt.quiver(
    X[:, 0],
    X[:, 1],
    Y[:, 0],
    Y[:, 1],
    angles="xy",
    scale_units="xy",
    width=0.001,
    scale=1,
    zorder=10,
)
plt.xlabel("$x$")
plt.ylabel("$y$")
# plt.axis('off')
plt.show()
