import gpflow as gpf
import matplotlib.pyplot as plt

from mogpe..data.utils import load_mixture_dataset, load_mcycle_dataset
from mogpe.models.mixture_model import init_fake_mixture
# from mogpe.models.trainer_new import GPMixtureTrainer
from mogpe.training.utils import monitored_training_loop
from mogpe.visualization.plotter import Plotter1D

epochs = 5000
batch_size = 40
# Load data set
data_file = '../../data/external/mcycle.csv'
# data, F, prob_a_0 = load_mixture_dataset(filename=data_file, standardise=False)
data = load_mcycle_dataset(filename=data_file)
X, Y = data
# plt.scatter(X, Y)
# plt.show()

model = init_fake_mixture(X, Y, num_experts=2, num_inducing_samples=1)

model.experts.experts_list[0].likelihood.variance.assign(1e-4)
gpf.set_trainable(model.experts.experts_list[0].likelihood.variance, False)

gpf.set_trainable(model.experts.experts_list[0].inducing_variable, False)
gpf.set_trainable(model.experts.experts_list[1].inducing_variable, False)
gpf.set_trainable(model.gating_network.inducing_variable, False)
gpf.utilities.print_summary(model)


# trainer.training_tf_loop(epochs=epochs, logging_epoch_freq=100)

def monitored_training_tf_loop(model,
                               training_loss,
                               epochs: int = 1,
                               num_batches_per_epoch: int = 1,
                               fast_tasks: gpf.monitor.MonitorTaskGroup = None,
                               logging_epoch_freq: int = 100):
monitored_training_tf_loop(epochs=epochs,
                                   fast_period=100,
                                   logging_epoch_freq=100)
# trainer.monitored_training_loop(epochs=epochs,
#                                 fast_period=100,
#                                 slow_period=500,
#                                 logging_epoch_freq=100)

gpf.utilities.print_summary(model)

plotter = Plotter1D(model, X, Y)

plotter.plot_model()
# plotter.plot_experts()
# plotter.plot_gating_netowrk()
# plt.show()
