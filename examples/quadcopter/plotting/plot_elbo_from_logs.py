#!/usr/bin/env python3
import numpy as np
import glob
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


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


def plot_training_curve(log_dir, fig=None, ax=None):
    filename = glob.glob(log_dir + "/events.out.tfevents.*")[0]
    ea = load_log_as_event_accumulator(filename, num_tensors=0)
    elbo_df = tensor_proto_to_ndarray(ea, "elbo")

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    sns.lineplot(data=elbo_df, x="step", y="elbo").set_title("ELBO")
    return fig, ax


if __name__ == "__main__":
    log_dir = "./logs/quadcopter/two_experts/08-17-181152"
    fig, ax = plot_training_curve(log_dir)
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
