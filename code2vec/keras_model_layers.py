"""
@author: Bohui Zhang

Custom tf.keras layers implemented in tensorflow 2.0 
Used in code2vec keras_model.py:
from keras_model_layers import Embedding, Concatenate, Dropout, TimeDistributed, Dense

Input layer has not been implemented yet

"""
import numpy as np

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer
# from tensorflow.keras.layers import InputLayer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers

from tensorflow.python.eager import context

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import generic_utils

from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import gen_math_ops

from tensorflow.python.ops.ragged import ragged_tensor


class Embedding(Layer):
    def __init__(self, input_dum, output_dim, 
                       embeddings_initializer='uniform', input_length=None, **kwargs):
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        dtype = kwargs.pop('dtype', K.floatx())
        kwargs['autocast'] = False
        super(Embedding, self).__init__(dtype=dtype, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.input_length = input_length

    def build(self, input_shape):
        if context.executing_eagerly() and context.context().num_gpus():
            with ops.device('cpu:0'):
                self.embeddings = self.add_weight(
                    name='embeddings',
                    shape=(self.input_dim, self.output_dim),
                    initializer='uniform',
                    trainable=True,
                    dtype=tf.float32
                )
        else:
            self.embeddings = self.add_weight(
                name='embeddings',
                shape=(self.input_dim, self.output_dim),
                initializer='uniform',
                trainable=True,
                dtype=tf.float32
            )

        self.built = True

    def call(self, inputs):
        dtype = K.dtype(inputs)
        if dtype != 'int32' and dtype != 'int64':
            inputs = math_ops.cast(inputs, 'int32')
        outputs = embedding_ops.embedding_lookup(self.embeddings, inputs)
        return outputs

class Concatenate(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(Concatenate, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        if all(shape is None for shape in input_shape):
            return
        reduced_inputs_shapes = [list(shape) for shape in input_shape]
        shape_set = set()
        for i in range(len(reduced_inputs_shapes)):
            del reduced_inputs_shapes[i][self.axis]
            shape_set.add(tuple(reduced_inputs_shapes[i]))

    def _merge_function(self, inputs):
        return K.concatenate(inputs, axis=self.axis) 

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

class TimeDistributed(Layer):
    def __init__(self, layer, **kwargs):
        super(TimeDistributed, self).__init__(layer, **kwargs)

        self._always_use_reshape = (
            layer_utils.is_builtin_layer(layer) and
            not getattr(layer, 'stateful', False)
        )

    def _get_shape_tuple(self, init_tuple, tensor, start_idx, int_shape=None):
        # replace all None in int_shape by K.shape
        if int_shape is None:
            int_shape = K.int_shape(tensor)[start_idx:]
        if not any(not s for s in int_shape):
            return init_tuple + tuple(int_shape)
        shape = K.shape(tensor)
        int_shape = list(int_shape)
        for i, s in enumerate(int_shape):
            if not s:
                int_shape[i] = shape[start_idx + i]
        return init_tuple + tuple(int_shape)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        child_input_shape = [input_shape[0]] + input_shape[2:]

        super(TimeDistributed, self).build(tuple(child_input_shape))

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        child_input_shape = tensor_shape.TensorShape([input_shape[0]] + input_shape[2:])
        child_output_shape = self.layer.compute_output_shape(child_input_shape)
        if not isinstance(child_output_shape, tensor_shape.TensorShape):
            child_output_shape = tensor_shape.TensorShape(child_output_shape)
        child_output_shape = child_output_shape.as_list()
        timesteps = input_shape[1]
        return tensor_shape.TensorShape([child_output_shape[0], timesteps] + child_output_shape[1:])

    def call(self, inputs, training=None, mask=None):
        kwargs = {}

        input_shape = K.int_shape(inputs)
        if input_shape[0] and not self._always_use_reshape:
            inputs, row_lengths = K.convert_inputs_if_ragged(inputs)
            is_ragged_input = row_lengths is not None

            # batch size matters, use rnn-based implementation
            def step(x, _):
                output = self.layer(x, **kwargs)
                return output, []

            _, outputs, _ = K.rnn(
                step,
                inputs,
                initial_states=[],
                input_length=row_lengths[0] if is_ragged_input else input_shape[1],
                mask=mask,
                unroll=False)
            y = K.maybe_convert_to_ragged(is_ragged_input, outputs, row_lengths)
        else:
            # No batch size specified, therefore the layer will be able
            # to process batches of any size.
            # We can go with reshape-based implementation for performance.
            if isinstance(inputs, ragged_tensor.RaggedTensor):
                y = self.layer(inputs.values, **kwargs)
                y = ragged_tensor.RaggedTensor.from_row_lengths(
                    y,
                    inputs.nested_row_lengths()[0])
            else:
                input_length = input_shape[1]
                if not input_length:
                    input_length = array_ops.shape(inputs)[1]
                    inner_input_shape = self._get_shape_tuple((-1,), inputs, 2)
                # Shape: (num_samples * timesteps, ...). And track the
                # transformation in self._input_map.
                inputs = array_ops.reshape(inputs, inner_input_shape)
                # (num_samples * timesteps, ...)
                if generic_utils.has_arg(self.layer.call, 'mask') and mask is not None:
                    inner_mask_shape = self._get_shape_tuple((-1,), mask, 2)
                    kwargs['mask'] = K.reshape(mask, inner_mask_shape)

                y = self.layer(inputs, **kwargs)

                # Shape: (num_samples, timesteps, ...)
                output_shape = self.compute_output_shape(input_shape).as_list()
                output_shape = self._get_shape_tuple((-1, input_length), y, 1, output_shape[2:])
                y = array_ops.reshape(y, output_shape)

        return y

class Dense(Layer):
    def __init__(self, units, activation=None, use_bias=True, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        dtype = dtype.as_dtype(self.dtype or K.floatx())
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])

        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim, self.units],
            dtype=self.dtype,
            trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units,],
                dtype=self.dtype,
                trainable=True
            )
        else:
            self.bias = None

        super(Dense, self).build(input_shape)

    def call(self, inputs):
        rank = inputs.shape.rank
        if rank is not None and rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            inputs = math_ops.cast(inputs, self._compute_dtype)
            if K.is_sparse(inputs):
                outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, self.kernel)
            else:
                outputs = gen_math_ops.mat_mul(inputs, self.kernel)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        # activation = 'softmax'
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs
