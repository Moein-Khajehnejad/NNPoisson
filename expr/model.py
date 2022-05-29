import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as layers, backend as K
from tensorflow.python.keras.constraints import NonNeg
from tensorflow.python.keras.models import Model
from util import DLogger
from util.types import Types

def abs_glorot_uniform(shape, dtype=None, partition_info=None):
    return K.abs(tf.keras.initializers.glorot_uniform(seed=None)(shape, dtype=dtype))


def build_model(conf_num, stim_shape, config):
    DLogger.logger().debug("Model created with configuration set" + str(conf_num))
    input_stim = layers.Input(shape=stim_shape)
    input_scaling = layers.Input(shape=(1,), dtype=Types.TF_FLOAT)
    spike_times = layers.Input(shape=(1,), dtype=Types.TF_FLOAT)
    l = 0
    stim_layer1 = layers.Dense(config['size_stim_layer1'], activation='softplus', dtype=Types.TF_FLOAT,
                               kernel_initializer=tf.keras.initializers.GlorotNormal(),
                               bias_initializer=tf.keras.initializers.RandomUniform(),
                               name=f'neuro{l}'
                               )(input_stim)
    l += 1
    stim_layer2 = layers.Dense(config['size_stim_layer2'], activation='softplus', dtype=Types.TF_FLOAT,
                               kernel_initializer=tf.keras.initializers.GlorotNormal(),
                               bias_initializer=tf.keras.initializers.RandomUniform(),
                               name=f'neuro{l}'
                               )(stim_layer1)
    l += 1
    stim_layer3 = layers.Dense(config['size_time_layer'], activation='linear', dtype=Types.TF_FLOAT,
                               kernel_initializer=tf.keras.initializers.GlorotNormal(),
                               bias_initializer=tf.keras.initializers.RandomUniform(),
                               name=f'neuro{l}'
                               )(stim_layer2)
    scaled_time_layer = tf.math.multiply(input_scaling, spike_times)

    # scaled_time_layer = layers.Lambda(lambda x: (K.log(x)))(scaled_time_layer)

    l += 1
    time_layer = layers.Dense(config['size_time_layer'], activation='linear', dtype=Types.TF_FLOAT,
                              kernel_initializer=abs_glorot_uniform,
                              kernel_constraint=NonNeg(), use_bias=False,
                              name=f'neuro{l}')(scaled_time_layer)

    StimTime_layer = layers.Lambda(lambda inputs: K.tanh(inputs[0] + inputs[1]), name='mix')([stim_layer3, time_layer])
    l += 1
    neural_layer1 = layers.Dense(config['size_neural_layer1'], activation='tanh', dtype=Types.TF_FLOAT,
                                 kernel_initializer=abs_glorot_uniform,
                                 kernel_constraint=NonNeg(),
                                 bias_initializer=tf.keras.initializers.RandomUniform(),
                                 name=f'neuro{l}')(StimTime_layer)

    l += 1
    RTlayer = layers.Dense(config['size_neural_layer1'], activation='linear', dtype=Types.TF_FLOAT,
                           kernel_initializer=abs_glorot_uniform,
                           name=f'neuro{l}')(input_stim[:, -1, np.newaxis])

    neural_layer1 = layers.Lambda(lambda inputs: K.tanh(inputs[0] + inputs[1]), name='mix2')([neural_layer1, RTlayer])

    l += 1
    neural_layer2 = layers.Dense(config['size_neural_layer2'], activation='softplus', dtype=Types.TF_FLOAT,
                                 kernel_initializer=abs_glorot_uniform,
                                 kernel_constraint=NonNeg(),
                                 bias_initializer=tf.keras.initializers.RandomUniform(),
                                 name=f'neuro{l}')(neural_layer1)
    nonScaled_neural_layer = neural_layer2 * (1 / input_scaling)
    # neural_to_beh_left = nonScaled_neural_layer
    l = 0
    beh_layer1_left = layers.Dense(config['size_beh_layer1'], activation='tanh', dtype=Types.TF_FLOAT,
                                   kernel_initializer=abs_glorot_uniform,
                                   kernel_constraint=NonNeg(),
                                   bias_initializer=tf.keras.initializers.RandomUniform(),
                                   name=f'beh{l}'
                                   )(neural_layer2)

    l += 1
    beh_layer2_left = layers.Dense(config['size_beh_layer2'], activation='softplus', dtype=Types.TF_FLOAT,
                                   kernel_initializer=abs_glorot_uniform,
                                   kernel_constraint=NonNeg(),
                                   bias_initializer=tf.keras.initializers.RandomUniform(),
                                   name=f'beh{l}')(beh_layer1_left)
    # neural_to_beh_right = nonScaled_neural_layer
    l += 1
    beh_layer1_right = layers.Dense(config['size_beh_layer1'], activation='tanh', dtype=Types.TF_FLOAT,
                                    kernel_initializer=abs_glorot_uniform,
                                    kernel_constraint=NonNeg(),
                                    bias_initializer=tf.keras.initializers.RandomUniform(),
                                    name=f'beh{l}')(neural_layer2)

    l += 1
    beh_layer2_right = layers.Dense(config['size_beh_layer2'], activation='softplus', dtype=Types.TF_FLOAT,
                                    kernel_initializer=abs_glorot_uniform,
                                    kernel_constraint=NonNeg(),
                                    bias_initializer=tf.keras.initializers.RandomUniform(),
                                    name=f'beh{l}')(beh_layer1_right)

    nonScaled_beh_layer2_left = beh_layer2_left * (1 / input_scaling)
    nonScaled_beh_layer2_right = beh_layer2_right * (1 / input_scaling)
    model = Model(inputs=[input_stim, spike_times, input_scaling],
                  outputs=[nonScaled_beh_layer2_left, nonScaled_beh_layer2_right, neural_layer2, nonScaled_neural_layer])

    return model