"""
@author: Bohui Zhang

Custom tf.keras layers implemented in tensorflow 2.0 
Used for code2vec keras_model.py:
from keras_model_layers import Input, Embedding, Concatenate, Dropout, TimeDistributed, Dense

"""
import numpy as np

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K

from tensorflow.python.framework import ops

from tensorflow.python.keras.utils import tf_utils

from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn

'''
class Input(Layer):
    def __init__(self, input_shape=None, batch_size=None, name=None, dtype=None, **kwargs):
        super(Input, self).__init__(input_shape, name=name, dtype=dtype, **kwargs)

    def build(self, inputs_shape):



        super(Input, self).build(inputs_shape)

    def call(self, inputs, **kwargs):

        return 
'''
class Embedding(Layer):
    def __init__(self, input_dum, output_dim, name=None, **kwargs):
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        super(Embedding).__init__(name=name, **kwargs)

    def build(self, input_shape):

        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer='uniform',
            trainable=True,
            dtype=tf.float32
        )

        super(Embedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        result = embedding_ops.embedding_lookup(self.embeddings, inputs)
        return result
'''
class Concatenate(Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(Concatenate).__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def build(self, inputs_shape):


        super(Concatenate, self).build(inputs_shape)

    def call(self, inputs, **kwargs):

        return 
'''

class Dropout(Layer):

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        #self.supports_masking = True

    def _get_noise_shape(self, inputs):

        if self.noise_shape is None:
            return self.noise_shape
        
        concrete_inputs_shape = array_ops.shape(inputs)
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(concrete_inputs_shape[i] if value is None else value)
        return ops.convert_to_tensor_v2(noise_shape)

    def call(self, inputs, training=None):

        def dropped_inputs():
            return nn.dropout(
                inputs,
                noise_shape=self._get_noise_shape(inputs),
                seed=self.seed,
                rate=self.rate
            )
        
        # Return either `dropped_inputs` if training is true else a Tensor with the same shape and contents as inputs
        output = tf_utils.smart_cond(training, dropped_inputs, lambda: array_ops.identity(inputs))

        return output
'''
class TimeDistributed(Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(TimeDistributed).__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def build(self, inputs_shape):


        super(TimeDistributed, self).build(inputs_shape)

    def call(self, inputs, **kwargs):

        return 

class Dense(Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(Dense).__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def build(self, inputs_shape):


        super(Dense, self).build(inputs_shape)

    def call(self, inputs, **kwargs):

        return 
'''