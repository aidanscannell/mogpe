import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time

from datetime import datetime
from typing import Tuple

from experts import ExpertsSeparate
from gating_network import GatingNetwork
from svmogpe import SVMoGPE
# from plotter import Plotter

from gpflow.ci_utils import ci_niter
from gpflow.config import default_float
from gpflow.utilities import print_summary
from gpflow.monitor import (ExecuteCallback, ImageToTensorBoard,
                            ModelToTensorBoard, Monitor, MonitorTaskGroup,
                            ScalarToTensorBoard, ToTensorBoard)


class Trainer:
    def __init__(self,
                 model,
                 dataset,
                 batch_size=5,
                 optimizer=None,
                 log_dir='',
                 log_dir_date=True):

        if log_dir_date:
            self.log_dir = log_dir + '/' + datetime.now().strftime(
                "%m-%d-%H%M%S")
        else:
            self.log_dir = log_dir
        ckpt_dir = self.log_dir + "/ckpt"
        # self.model_save_dir = self.log_dir + "/saved_model"

        self.X, self.Y = dataset
        x_min = self.X.numpy().min() * 1.2
        x_max = self.X.numpy().max() * 1.2
        self.input = np.linspace(x_min, x_max, 100).reshape(-1, 1)

        num_train_data = self.X.shape[0]
        # self.batch_size = batch_size
        minibatch_size = batch_size
        self.num_batches_per_epoch = num_train_data // batch_size

        self.model = model

        if optimizer is None:
            self.optimizer = tf.optimizers.Adam()
        else:
            self.optimizer = optimizer

        train_dataset = tf.data.Dataset.from_tensor_slices(
            dataset).repeat().shuffle(num_train_data)
        train_iter = iter(train_dataset.batch(minibatch_size))
        self.training_loss = model.training_loss_closure(
            train_iter,
            compile=True)  # compile=True (default): compiles using tf.function

        # step_var = tf.Variable(1, dtype=tf.int32, trainable=False)
        self.step_var = None
        self.epoch_var = tf.Variable(1, dtype=tf.int32, trainable=False)
        ckpt = tf.train.Checkpoint(
            model=self.model,
            # step=self.step_var,
            epoch=self.epoch_var)
        self.manager = tf.train.CheckpointManager(ckpt,
                                                  ckpt_dir,
                                                  max_to_keep=5)

    def optimization_step(self):
        self.optimizer.minimize(self.training_loss,
                                self.model.trainable_variables)

    def simple_training_loop(self,
                             epochs: int = 1,
                             logging_epoch_freq: int = 10):
        tf_optimization_step = tf.function(self.optimization_step)

        for epoch in range(epochs):
            tf_optimization_step()

            epoch_id = epoch + 1
            if epoch_id % logging_epoch_freq == 0:
                tf.print(f"Epoch {epoch_id}")

    def checkpointing_training_loop(self,
                                    epochs: int,
                                    logging_epoch_freq: int = 100):
        tf_optimization_step = tf.function(self.optimization_step)

        for epoch in range(epochs):
            tf_optimization_step()
            if self.epoch_var is not None:
                self.epoch_var.assign(epoch + 1)

            epoch_id = epoch + 1
            if epoch_id % logging_epoch_freq == 0:
                ckpt_path = self.manager.save()
                tf.print(f"Epoch {epoch_id}")

    def init_fast_tasks(self, fast_period):
        elbo_task = ScalarToTensorBoard(self.log_dir, self.training_loss,
                                        "elbo")
        model_task = ModelToTensorBoard(self.log_dir, self.model)
        return MonitorTaskGroup([model_task, elbo_task], period=fast_period)

    def init_slow_tasks(self, slow_period):
        image_task_y = ImageToTensorBoard(self.log_dir, self.plot_model_y,
                                          "plot_model_y")
        image_task_prob_a_0 = ImageToTensorBoard(self.log_dir,
                                                 self.plot_model_prob_a_0,
                                                 "plot_model_prob_a_0")
        image_task_f1 = ImageToTensorBoard(self.log_dir,
                                           self.plot_model_expert_1_f,
                                           "plot_model_expert_1_f")
        image_task_f2 = ImageToTensorBoard(self.log_dir,
                                           self.plot_model_expert_2_f,
                                           "plot_model_expert_2_f")
        image_task_y1 = ImageToTensorBoard(self.log_dir,
                                           self.plot_model_expert_1_y,
                                           "plot_model_expert_1_y")
        image_task_y2 = ImageToTensorBoard(self.log_dir,
                                           self.plot_model_expert_2_y,
                                           "plot_model_expert_2_y")
        # image_task_y_samples = ImageToTensorBoard(log_dir,
        #                                           self.plot_y_samples,
        #                                           "plot_model_y_samples")

        image_tasks = [
            image_task_y, image_task_f1, image_task_f2, image_task_y1,
            image_task_y2, image_task_prob_a_0
            # , image_task_y_samples
        ]
        return MonitorTaskGroup(image_tasks, period=slow_period)

    def monitor_training_tf_loop(self,
                                 fast_period,
                                 epochs: int,
                                 logging_epoch_freq: int = 100):
        self.init_monitor(fast_period)

        @tf.function
        def monitored_opt_step():
            self.optimization_step()
            self.monitor(epoch)

        t = time.time()
        iterations = ci_niter(epochs)
        for epoch in tf.range(iterations):
            monitored_opt_step()
            # epoch_id = epoch + 1
            # if epoch_id % logging_epoch_freq == 0:
            #     tf.print(f"Epoch {epoch_id}")
            duration = t - time.time()
            print("Iteration duration: ", duration)
            t = time.time()

    def monitor_training_loop(self,
                              epochs: int,
                              fast_period,
                              slow_period=None,
                              logging_epoch_freq: int = 100):
        fast_tasks = self.init_fast_tasks(fast_period)
        if slow_period is None:
            self.monitor = Monitor(fast_tasks)
        else:
            slow_tasks = self.init_slow_tasks(slow_period)
            self.monitor = Monitor(fast_tasks, slow_tasks)

        tf_optimization_step = tf.function(self.optimization_step)

        t = time.time()
        for epoch in range(epochs):
            tf_optimization_step()
            self.monitor(epoch)

            epoch_id = epoch + 1
            if epoch_id % logging_epoch_freq == 0:
                tf.print(f"Epoch {epoch_id}")

            # duration = t - time.time()
            # print("Iteration duration: ", duration)
            # t = time.time()

    def monitor_and_ckpt_training_loop(self,
                                       fast_period,
                                       epochs: int,
                                       saving_epoch_freq: int = 2000,
                                       logging_epoch_freq: int = 100):
        self.init_monitor(fast_period)
        tf_optimization_step = tf.function(self.optimization_step)

        # t = time.time()
        for epoch in range(epochs):
            tf_optimization_step()
            self.monitor(epoch)
            if self.epoch_var is not None:
                self.epoch_var.assign(epoch + 1)

            epoch_id = epoch + 1
            if epoch_id % logging_epoch_freq == 0:
                ckpt_path = self.manager.save()
                tf.print(f"Epoch {epoch_id}")

    def plot_gp(self, fig, ax, mean, var):
        alpha = 0.4
        ax.scatter(self.X, self.Y, marker='x', color='k', alpha=alpha)
        ax.plot(self.input, mean, "C0", lw=2)
        ax.fill_between(
            self.input[:, 0],
            mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
            mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
            color="C0",
            alpha=0.2,
        )

    def plot_model_expert_1_f(self, fig, ax):
        tf.print("Plotting f1...")
        f_means, f_vars = self.model.predict_experts_fs(self.input)
        self.plot_gp(fig, ax, f_means[0], f_vars[0])

    def plot_model_expert_2_f(self, fig, ax):
        tf.print("Plotting f2...")
        f_means, f_vars = self.model.predict_experts_fs(self.input)
        self.plot_gp(fig, ax, f_means[1], f_vars[1])

    def plot_model_prob_a_0(self, fig, ax):
        tf.print("Plotting mixing prob ...")
        prob_a_0, _ = self.model.predict_mixing_probs(self.input)
        ax.plot(self.input, prob_a_0)

    def plot_model_expert_1_y(self, fig, ax):
        tf.print("Plotting y1...")
        y_means, y_vars = self.model.predict_experts_ys(self.input)
        self.plot_gp(fig, ax, y_means[0], y_vars[0])

    def plot_model_expert_2_y(self, fig, ax):
        tf.print("Plotting y2...")
        y_means, y_vars = self.model.predict_experts_ys(self.input)
        self.plot_gp(fig, ax, y_means[1], y_vars[1])

    def plot_model_y(self, fig, ax):
        tf.print("Plotting y...")
        y_mean, y_var = self.model.predict_y_moment_matched(self.input)
        self.plot_gp(fig, ax, y_mean, y_var)


# if __name__ == "__main__":
#     # dataset config
#     dataset_name = 'mcycle'
#     # dataset_name = 'sin'

#     # inference config
#     # batch_size = 5
#     # batch_size = 30
#     batch_size = 50
#     batch_size = 100
#     # minibatch_size = 100
#     num_epochs = 15000
#     num_epochs = 20000
#     # num_epochs = 5000
#     bound = 'tight'
#     # bound = 'titsias'
#     num_inducing = 80
#     num_inducing = None
#     num_inducing = 40
#     num_inducing = 5
#     num_samples_expert_expectation = None
#     # num_samples_expert_expectation = 1

#     # monitoring config
#     fast_period = 10
#     slow_period = 1000
#     plot_images = True
#     log_dir_date = True
#     # log_dir_date = False

#     if num_samples_expert_expectation is None:
#         expert_expectation = 'analytic'
#     else:
#         expert_expectation = 'sample'
#     log_dir = "svmogpe/" + dataset_name + "/" + bound + "/" + expert_expectation + "-f/batch_size-" + str(
#         batch_size) + "/num_inducing-" + str(num_inducing) + "/"
#     save_model_dir = log_dir + "saved_model"

#     data = load_mcycle_dataset(filename='~/Developer/datasets/mcycle.csv')

#     # data, F, prob_a_0 = load_mixture_dataset(
#     #     filename=
#     #     '../data/artificial/artificial-1d-mixture-sin-gating-sin-expert-higher-noise.npz',
#     #     # filename='../data/artificial/artificial-1d-mixture-sin-gating.npz',
#     #     standardise=False)

#     X, Y = data
#     m = init_svmogpe(X, Y, num_inducing, bound, num_samples_expert_expectation)

#     # gpf.set_trainable(m.experts.experts[0].likelihood, False)
#     gpf.set_trainable(m.experts.experts[0].likelihood, True)
#     gpf.set_trainable(m.experts.experts[1].likelihood, True)

#     # gpf.set_trainable(m.gating_network.inducing_variable, True)
#     # gpf.set_trainable(m.experts.experts[0].inducing_variable, True)
#     # gpf.set_trainable(m.experts.experts[1].inducing_variable, True)
#     # gpf.set_trainable(m.gating_network.inducing_variable, False)
#     # gpf.set_trainable(m.experts.experts[0].inducing_variable, False)
#     # gpf.set_trainable(m.experts.experts[1].inducing_variable, False)

#     print_summary(m)

#     trainer = Trainer(m,
#                       data,
#                       batch_size=batch_size,
#                       log_dir=log_dir,
#                       log_dir_date=log_dir_date)

#     # trainer.simple_training_loop(epochs=num_epochs, logging_epoch_freq=10)
#     # trainer.checkpointing_training_loop(epochs=num_epochs,
#     #                                     logging_epoch_freq=100)
#     trainer.monitor_training_loop(epochs=num_epochs,
#                                   fast_period=fast_period,
#                                   slow_period=slow_period,
#                                   logging_epoch_freq=100)

#     # trainer.monitor_training_tf_loop(fast_period=fast_period,
#     #                                  epochs=num_epochs,
#     #                                  logging_epoch_freq=100)
#     # trainer.monitor_and_ckpt_training_loop(fast_period=fast_period,
#     #                                        epochs=num_epochs,
#     #                                        logging_epoch_freq=100)

#     # x_min = X.numpy().min() * 1.2
#     # x_max = X.numpy().max() * 1.2
#     # test_input = np.linspace(x_min, x_max, 100).reshape(-1, 1)
#     # plotter = Plotter(m, X.numpy(), Y.numpy(), test_input)
#     # plotter.plot_y_moment_matched()
#     # plotter.plot_ys()
#     # plt.show()
#     # save_model_dir = './saved_model_rbf_gating_kernel_rbf_expert'
#     # save_model_dir = './saved_model_svmogpe_mcycle'
#     # save_model(m, save_model_dir)
#     print_summary(m)
