#!/usr/bin/env python3
import numpy as np
import glob
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

plt.style.use("seaborn-paper")


def tensor_proto_to_ndarray(ea, name, stack=True):
    df = pd.DataFrame(ea.Tensors(name))
    if stack:
        df[name] = np.stack(list(map(tf.make_ndarray, df["tensor_proto"])), -1)
    else:
        df[name] = list(map(tf.make_ndarray, df["tensor_proto"]))
    return df


def load_log_as_event_accumulator(filename, num_tensors=0):
    return event_accumulator.EventAccumulator(
        filename,
        size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            # event_accumulator.IMAGES: 4,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.TENSORS: num_tensors,
            event_accumulator.HISTOGRAMS: 1,
        },
    ).Reload()


def plot_training_curve(log_dir, fig=None, ax=None, y_lim=None, title=None, label=None):
    filename = glob.glob(log_dir + "/events.out.tfevents.*")[0]
    ea = load_log_as_event_accumulator(filename, num_tensors=0)
    # elbo_df = tensor_proto_to_ndarray(ea, "elbo")
    elbo_df = tensor_proto_to_ndarray(ea, "training_loss")

    # kernel_size = 15
    kernel_size = 50
    kernel = np.ones(kernel_size) / kernel_size
    elbo_df["training_loss"] = np.convolve(
        elbo_df["training_loss"], kernel, mode="same"
    )

    if fig is None:
        # fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        fig, ax = plt.subplots(1, 1)
    # sns.lineplot(data=elbo_df, x="step", y="elbo").set_title("ELBO")
    # sns.lineplot(data=elbo_df, x="step", y="training_loss").set_title(title)
    ax.plot(elbo_df["step"], elbo_df["training_loss"], label=label)
    ax.set_title(title)
    ax.set(xlabel="step", ylabel="Negative ELBO")
    if y_lim is not None:
        # # ax.set_ylim(-250.0, 500.0)
        # ax.set_ylim(20.0, 150.0)
        ax.set_ylim(*y_lim)
    return fig, ax


def plot_training_curves(log_dirs: list, fig=None, ax=None, y_lim=None, save_dir=None):
    if fig is None:
        # fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        fig, ax = plt.subplots(1, 1)
    for log_dir in log_dirs:
        batch_size = log_dir.split("batch_size_")[1].split("/")[0]
        num_inducing = log_dir.split("num_inducing_")[1].split("/")[0]
        print("batch_size: " + batch_size)
        print("num_inducing: " + num_inducing)

        plot_training_curve(
            log_dir,
            fig,
            ax,
            y_lim=y_lim,
            label="$N_b=" + batch_size + "$",
            title="$M=" + num_inducing + "$",
        )
    ax.legend()
    if save_dir is not None:
        save_name = save_dir + "training_loss_num_ind_" + str(num_inducing)
        plt.savefig(save_name, transparent=True)
    return fig, ax


if __name__ == "__main__":
    # log_dir = "./logs/quadcopter/two_experts/08-17-181152"
    # fig, ax = plot_training_curve(log_dir)
    #
    log_dir = "./logs/mcycle/2_experts/batch_size_64/learning_rate_0.01/further_bound/num_inducing_133/10-07-162743"
    log_dirs = [
        "./logs/mcycle/2_experts/batch_size_32/learning_rate_0.01/further_bound/num_inducing_133/10-07-165652",
        "./logs/mcycle/2_experts/batch_size_32/learning_rate_0.001/further_bound/num_inducing_133/10-07-170805",
        "./logs/mcycle/2_experts/batch_size_64/learning_rate_0.01/further_bound/num_inducing_133/10-07-162743",
        "./logs/mcycle/2_experts/batch_size_133/learning_rate_0.01/further_bound/num_inducing_133/10-07-161617",
    ]

    # y_lim = (-250.0, 500.0)
    y_lim = (-50.0, 150.0)
    fig, ax = plot_training_curves(
        log_dirs, y_lim=y_lim, save_dir="./mcycle/images/training_curves/"
    )

    log_dirs = [
        "./logs/mcycle/2_experts/batch_size_16/learning_rate_0.01/further_bound/num_inducing_64/10-07-184520",
        "./logs/mcycle/2_experts/batch_size_32/learning_rate_0.01/further_bound/num_inducing_64/10-07-172303",
        "./logs/mcycle/2_experts/batch_size_64/learning_rate_0.01/further_bound/num_inducing_64/10-07-172221",
        "./logs/mcycle/2_experts/batch_size_133/learning_rate_0.01/further_bound/num_inducing_64/10-07-172134",
    ]
    # y_lim = (20.0, 150.0)
    fig, ax = plot_training_curves(
        log_dirs, y_lim=y_lim, save_dir="./mcycle/images/training_curves/"
    )

    log_dirs = [
        "./logs/mcycle/2_experts/batch_size_16/learning_rate_0.01/further_bound/num_inducing_32/10-07-185615",
        "./logs/mcycle/2_experts/batch_size_32/learning_rate_0.01/further_bound/num_inducing_32/10-07-185549",
        "./logs/mcycle/2_experts/batch_size_64/learning_rate_0.01/further_bound/num_inducing_32/10-07-184712",
        "./logs/mcycle/2_experts/batch_size_133/learning_rate_0.01/further_bound/num_inducing_32/10-07-184600",
    ]
    fig, ax = plot_training_curves(
        log_dirs, y_lim=y_lim, save_dir="./mcycle/images/training_curves/"
    )

    log_dirs = [
        "./logs/mcycle/2_experts/batch_size_16/learning_rate_0.01/further_bound/num_inducing_16/10-07-194718",
        "./logs/mcycle/2_experts/batch_size_32/learning_rate_0.01/further_bound/num_inducing_16/10-07-194635",
        "./logs/mcycle/2_experts/batch_size_64/learning_rate_0.01/further_bound/num_inducing_16/10-07-194606",
        "./logs/mcycle/2_experts/batch_size_133/learning_rate_0.01/further_bound/num_inducing_16/10-07-194531",
    ]

    fig, ax = plot_training_curves(
        log_dirs, y_lim=y_lim, save_dir="./mcycle/images/training_curves/"
    )
    plt.show()

# print(experts_df["experts_latent_function_posterior"][0])
# print(np.array(experts_df["experts_latent_function_posterior"][0]))
# print(experts_df["experts_latent_function_posterior"][0].shape)
# import io
# png_buffer = BytesIO()
# name = "experts_latent_function_posterior"
# image = experts_df["experts_latent_function_posterior"][0]
# image = tf.make_ndarray(pd.DataFrame(ea.Tensors(name))["tensor_proto"][0])
# image = pd.DataFrame(ea.Tensors(name))["tensor_proto"][0]
# image = tf.io.decode_proto(
#     image, message_type, field_names, output_types,
#     descriptor_source='local://', message_format='binary',
#     sanitize=False, name=None
# )
# print("imag")
# print(type(image))
# print(dir(image))
# print(image.string_val)
# print(type(image.string_val))
# image_tensor = tf.io.decode_png(png_encoded)[None]
# png = tf.io.decode_png(image, channels=3)
# png = tf.io.encode_png(image, compression=-1, name=None)
# print(png)
# print(type(png))

# stream_str = io.BytesIO(b"JournalDev Python: x00x01")
# print(stream_str.getvalue())

# from PIL import Image


# image2 = Image.fromarray(experts_df["experts_latent_function_posterior"][0])
# print(type(image2))
# print(image2.mode)
# print(image2.size)

# with open("image.png", "w") as output:
#     output.write(experts_df["experts_latent_function_posterior"][0])
