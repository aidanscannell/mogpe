import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp
from gpflow.utilities import print_summary

tfd = tfp.distributions


def init_gaus_dists(means, vars):
    dists = []
    for mean, var in zip(means, vars):
        dist = tfd.Normal(loc=mean, scale=var)
        dists.append(dist)
    return dists


def gp(x_train, y_train, x_test):
    kernel = gpf.kernels.RBF()
    m = gpf.models.GPR((x_train, y_train), kernel=kernel)
    opt = gpf.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss,
                            m.trainable_variables,
                            options=dict(maxiter=1000))
    print_summary(m)
    return m.predict_f(x_test)


def init_subplot():
    x1 = np.linspace(-9, -4, 100)
    x2 = np.linspace(-4, 4, 100)
    x3 = np.linspace(4, 9, 100)

    params = {
        'axes.labelsize': 20,
        'font.size': 20,
        'legend.fontsize': 20,
        # 'legend.fontsize': 10,
        # 'xtick.labelsize': 28,
        # 'ytick.labelsize': 28,
        'text.usetex': True,
        # 'figure.figsize': [12, 4]
    }
    plt.rcParams.update(params)

    fig, axs = plt.subplots(nrows=2,
                            ncols=3,
                            figsize=(14.5, 5.5),
                            sharex='all')
    for i in range(2):
        for j in range(3):
            if i != 0 or j != 0:
                axs[i, j].spines['top'].set_visible(False)
                axs[i, j].spines['right'].set_visible(False)
                # for spine in axs[i, j].spines.values():
                #     spine.set_position(('outward', 5))
                axs[i, j].fill_between(x1.reshape(-1),
                                       0,
                                       1,
                                       color=color_1,
                                       alpha=0.2)
                axs[i, j].fill_between(x2.reshape(-1),
                                       0,
                                       1,
                                       color=color_2,
                                       alpha=0.2)
                axs[i, j].fill_between(x3.reshape(-1),
                                       0,
                                       1,
                                       color=color_1,
                                       alpha=0.2)
            else:
                axs[i, j].spines['top'].set_visible(False)
                axs[i, j].spines['right'].set_visible(False)
                axs[i, j].spines['left'].set_visible(False)
                axs[i, j].get_yaxis().set_visible(False)
                axs[i, j].spines['bottom'].set_visible(False)
                axs[i, j].get_xaxis().set_visible(False)

    axs[1, 0].set_xlabel('$x$')
    axs[1, 1].set_xlabel('$x$')
    axs[1, 2].set_xlabel('$x$')
    axs[1, 0].set_ylabel('$\Pr(\\alpha=k | x)$')
    axs[1, 1].set_ylabel('$\Pr(\\alpha=k | x)$')
    axs[1, 2].set_ylabel('$\Pr(\\alpha=k | x)$')
    axs[0, 1].set_ylabel('$p(x| \\alpha=k, z=c )$')
    axs[0, 2].set_ylabel('$p(x | \\alpha=k)$')
    # axs[2, 1].set_ylabel('$p(x | \\alpha=k )$')
    # axs[0, 0].get_xaxis().set_visible(False)
    # axs[1, 0].get_xaxis().set_visible(False)
    # axs[1, 1].get_xaxis().set_visible(False)
    # axs[0, 0].spines['bottom'].set_visible(False)
    # axs[1, 0].spines['bottom'].set_visible(False)
    # axs[1, 1].spines['bottom'].set_visible(False)
    return fig, axs


color_1 = 'olive'
color_2 = 'darkslategrey'
color_3 = 'darkred'

num_data = 3000
x_test = np.linspace(-9, 9, num_data).reshape(num_data, 1)

########################################################
# Create gating network based on GPs
########################################################
y_train = np.array([0., 0., 0.1, 0.5, 0.9, 1., 1., 0.9, 0.5, 0.1, 0.,
                    0.]).reshape(-1, 1)
num_train = y_train.shape[0]
x_train = np.linspace(-9, 9, num_train).reshape(-1, 1)
gp_prob_a_0, _ = gp(x_train, y_train, x_test)
gp_prob_a_1 = -gp_prob_a_0 + 1

########################################################
# Create gating network based on GMM with expert
# indicator = cluster indicator
########################################################
# Create 3 Gaussians (one cluster per expert)
means = [-6., 0., 6.]
vars = [0.4, 0.8, 0.4]
vars = [1.4, 1.8, 1.4]
dists = init_gaus_dists(means, vars)

# Create two mixtures of 3 Gaussians
probs_mix1 = [0.5, 0.1, 0.4]
mixture_1 = tfd.Mixture(cat=tfd.Categorical(probs=probs_mix1),
                        components=dists)
probs_mix2 = [0.1, 0.8, 0.1]
mixture_2 = tfd.Mixture(cat=tfd.Categorical(probs=probs_mix2),
                        components=dists)

# We assume there are 3 Gaussian clusters (one for each expert)
px_given_experts = [mix.prob(x_test).numpy() for mix in dists]
p_experts = [0.7, 0.2, 0.1]

# Calculate the evidence and joint prob of x and expert indicator (needed for Baye's rule)
px = 0
joint_px_and_experts = []
for prob_expert, px_given_expert in zip(p_experts, px_given_experts):
    joint_prob = prob_expert * px_given_expert
    px += joint_prob
    joint_px_and_experts.append(joint_prob)

# Apply Baye's rule to get prob of expert indicator given inputs
p_experts_given_x = []
for joint_px_and_expert in joint_px_and_experts:
    p_experts_given_x.append(joint_px_and_expert / px)

########################################################
# Create gating network based on GMM with expert
# indicator != cluster indicator (2 experts and 6 clusters)
########################################################
# Now lets assume 2 experts and 3 clusters with separate indicators
means = [means[0], means[1], means[2], means[0], means[1], means[2]]
vars = [vars[0], vars[1], vars[2], vars[0], vars[1], vars[2]]
dists = init_gaus_dists(means, vars)
prob_expert1 = 0.8
prob_expert2 = 0.2
px_given_e1c1 = dists[0].prob(x_test).numpy()
px_given_e1c2 = dists[1].prob(x_test).numpy()
px_given_e1c3 = dists[2].prob(x_test).numpy()
px_given_e2c1 = dists[3].prob(x_test).numpy()
px_given_e2c2 = dists[4].prob(x_test).numpy()
px_given_e2c3 = dists[5].prob(x_test).numpy()

# marginalise cluster indicator c
px_given_expert1_and_cluster_cs = [px_given_e1c1, px_given_e1c2, px_given_e1c3]
px_given_expert2_and_cluster_cs = [px_given_e2c1, px_given_e2c2, px_given_e2c3]
prob_cluster_cs_given_expert_1 = [0.8, 0.3, 0.9]
prob_cluster_cs_given_expert_2 = [0.2, 0.7, 0.1]
# prob_cluster_cs_given_expert_1 = [0.4, 0.1, 0.4]
# prob_cluster_cs_given_expert_2 = [0.1, 0.8, 0.1]
px_given_expert1 = 0
joint_px_and_clusters_given_expert1 = []
for px_given_expert1_and_cluster_c, prob_cluster_c in zip(
        px_given_expert1_and_cluster_cs, prob_cluster_cs_given_expert_1):
    joint_px_and_cluster_given_expert1 = px_given_expert1_and_cluster_c * prob_cluster_c
    px_given_expert1 += joint_px_and_cluster_given_expert1
    joint_px_and_clusters_given_expert1.append(
        joint_px_and_cluster_given_expert1)
px_given_expert2 = 0
joint_px_and_clusters_given_expert2 = []
for px_given_expert2_and_cluster_c, prob_cluster_c in zip(
        px_given_expert2_and_cluster_cs, prob_cluster_cs_given_expert_2):
    joint_px_and_cluster_given_expert2 = px_given_expert2_and_cluster_c * prob_cluster_c
    px_given_expert2 += joint_px_and_cluster_given_expert2
    joint_px_and_clusters_given_expert2.append(
        joint_px_and_cluster_given_expert2)

px = px_given_expert1 * prob_expert1 + px_given_expert2 * prob_expert2
prob_expert1_given_x = joint_px_and_cluster_given_expert1 / px
prob_expert2_given_x = joint_px_and_cluster_given_expert2 / px
# prob_expert1_given_x = (px_given_e1c1 * pc1_given_e1 + px_given_e1c2 * pc2_given_e1 +
#                px_given_e1c3 * pc3_given_e1) / px
# prob_expert2_given_x = (px_given_e2c1 * pc1_given_e2 + px_given_e2c2 * pc2_given_e2 +
#                px_given_e2c3 * pc3_given_e2) / px
normaliser = prob_expert1_given_x + prob_expert2_given_x
# prob_expert1_given_x /= normaliser
# prob_expert2_given_x /= normaliser

pe1 = 0.8
pe2 = 0.2
px = px_given_expert1 * pe1 + px_given_expert2 * pe2
pe1_given_x = (px_given_e1c1 * prob_cluster_cs_given_expert_1[0] +
               px_given_e1c2 * prob_cluster_cs_given_expert_1[1] +
               px_given_e1c3 * prob_cluster_cs_given_expert_1[2]) / px
pe2_given_x = (px_given_e2c1 * prob_cluster_cs_given_expert_2[0] +
               px_given_e2c2 * prob_cluster_cs_given_expert_2[1] +
               px_given_e2c3 * prob_cluster_cs_given_expert_2[2]) / px
# pe1_given_x = (px_given_e1c1 * pc1_given_e1 + px_given_e1c2 * pc2_given_e1 +
#                px_given_e1c3 * pc3_given_e1) / px
# pe2_given_x = (px_given_e2c1 * pc1_given_e2 + px_given_e2c2 * pc2_given_e2 +
#                px_given_e2c3 * pc3_given_e2) / px
normaliser = pe1_given_x + pe2_given_x
pe1_given_x /= normaliser
pe2_given_x /= normaliser

# plt.plot(x_test, joint_px_and_clusters_given_expert1[0])
# plt.plot(x_test, joint_px_and_clusters_given_expert1[1])
# plt.plot(x_test, joint_px_and_clusters_given_expert1[2])
# plt.plot(x_test, joint_px_and_clusters_given_expert2[0])
# plt.plot(x_test, joint_px_and_clusters_given_expert2[1])
# plt.plot(x_test, joint_px_and_clusters_given_expert2[2])
# plt.show()

########################################################
# Lets plot it all
########################################################
colors = [color_1, color_2, color_3]
linestyles = ['-.', '-', ':']
fig, axs = init_subplot()
axs[1, 0].plot(x_test, gp_prob_a_0, color=color_1, linestyle=linestyles[0])
axs[1, 0].plot(x_test, gp_prob_a_1, color=color_2, linestyle=linestyles[1])

for k, (p_expert_given_x, px_given_expert, color, linestyle) in enumerate(
        zip(p_experts_given_x, px_given_experts, colors, linestyles)):
    label = 'k=' + str(k + 1)
    axs[1, 1].plot(x_test,
                   p_expert_given_x,
                   color=color,
                   linestyle=linestyle,
                   label=label)
    axs[0, 1].plot(x_test, px_given_expert, color=color, linestyle=linestyle)

axs[1, 2].plot(
    x_test,
    # prob_expert1_given_x,
    pe1_given_x,
    # label='$P(\\alpha=1 | \mathbf{x})$',
    linestyle=linestyles[0],
    color=color_1)
axs[1, 2].plot(
    x_test,
    # prob_expert2_given_x,
    pe2_given_x,
    # label='$P(\\alpha=2 | \mathbf{x})$',
    linestyle=linestyles[1],
    color=color_2)
axs[0, 2].plot(
    x_test,
    px_given_expert1,
    # label='$P(\mathbf{x} | \\alpha=1)$',
    linestyle=linestyles[0],
    color=color_1)
axs[0, 2].plot(
    x_test,
    px_given_expert2,
    # label='$P(\mathbf{x} | \\alpha=2)$',
    linestyle=linestyles[1],
    color=color_2)
px_given_experts = np.array(px_given_experts)
limits = [0, px_given_experts.max() * 1.2]
axs[0, 1].set_ylim(limits)
axs[0, 2].set_ylim(limits)

fig.legend(loc='upper left', bbox_to_anchor=(0.13, 0.85))
# plt.savefig("../../references/images/gating-network-comparison-2by3.pdf",
#             transparent=True)
plt.show()
