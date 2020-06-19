"""
@author: Bohui Zhang

Custom layers model implemented in tensorflow 2.0 via subclassing.
To use in code2vec by substituting import info, `__init__` & `_create_keras_model`
functions and adding `call` function in `Code2VecModel` class in `keras_model.py`.


"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Concatenate, Dropout, TimeDistributed, Dense
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import sparse_top_k_categorical_accuracy

from path_context_reader import PathContextReader, ModelInputTensorsFormer, ReaderInputTensors, EstimatorAction
import os
import numpy as np
from functools import partial
from typing import List, Optional, Iterable, Union, Callable, Dict
from collections import namedtuple
import time
import datetime
from vocabularies import VocabType, Code2VecVocabs
from keras_attention_layer import AttentionLayer
from keras_topk_word_predictions_layer import TopKWordPredictionsLayer
from keras_words_subtoken_metrics import WordsSubtokenPrecisionMetric, WordsSubtokenRecallMetric, WordsSubtokenF1Metric
from config import Config
from common import common
from model_base import Code2VecModelBase, ModelEvaluationResults, ModelPredictionResults
from keras_checkpoint_saver_callback import ModelTrainingStatus, ModelTrainingStatusTrackerCallback,\
    ModelCheckpointSaverCallback, MultiBatchCallback, ModelTrainingProgressLoggerCallback


class Code2VecModel(Code2VecModelBase):
    def __init__(self, config: Config):
        self.keras_train_model: Optional[keras.Model] = None
        self.keras_eval_model: Optional[keras.Model] = None
        self.keras_model_predict_function: Optional[K.GraphExecutionFunction] = None
        self.training_status: ModelTrainingStatus = ModelTrainingStatus()
        self._checkpoint: Optional[tf.train.Checkpoint] = None
        self._checkpoint_manager: Optional[tf.train.CheckpointManager] = None

        # TODO: verify the initialization here
        self.config = config
        self.vocabs = Code2VecVocabs(self.config)

        # initialize the layers
        self.paths_embedding = Embedding(self.vocabs.path_vocab.size, self.config.PATH_EMBEDDINGS_SIZE, name='path_embedding')
        self.token_embedding = Embedding(self.vocabs.token_vocab.size, self.config.TOKEN_EMBEDDINGS_SIZE, name='token_embedding')
        self.context_embedding = Concatenate()
        self.context_dropout = Dropout(1 - self.config.DROPOUT_KEEP_RATE)
        self.context_dense = TimeDistributed(Dense(self.config.CODE_VECTOR_SIZE, use_bias=False, activation='tanh'))
        self.attention = AttentionLayer(name='attention')
        self.output = Dense(self.vocabs.target_vocab.size, use_bias=False, activation='softmax', name='target_index')
        self.topk = TopKWordPredictionsLayer(self.config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION,
                                             self.vocabs.target_vocab.get_index_to_word_lookup_table(), name='target_string')

        super(Code2VecModel, self).__init__(config)

    def call(self, inputs):
        path_source_token_input, path_input, path_target_token_input, context_valid_mask = inputs

        # Input paths are indexes, we embed these here.
        paths_embedded = self.paths_embedding(path_input)

        # Input terminals are indexes, we embed these here.
        path_source_token_embedded = self.token_embedding(path_source_token_input)
        path_target_token_embedded = self.token_embedding(path_target_token_input)

        # `Context` is a concatenation of the 2 terminals & path embedding.
        # Each context is a vector of size 3 * EMBEDDINGS_SIZE.
        context_embedded = self.context_embedding([path_source_token_embedded, paths_embedded, path_target_token_embedded])
        context_embedded = self.context_dropout(context_embedded)

        # Lets get dense: Apply a dense layer for each context vector (using same weights for all of the context).
        context_after_dense = self.context_dense(context_embedded)

        # The final code vectors are received by applying attention to the "densed" context vectors.
        code_vectors, attention_weights = self.attention([context_after_dense, context_valid_mask])

        # "Decode": Now we use another dense layer to get the target word embedding from each code vector.
        target_index = self.output(code_vectors)

        return code_vectors, attention_weights, target_index

    def _create_keras_model(self):
        # Each input sample consists of a bag of x`MAX_CONTEXTS` tuples (source_terminal, path, target_terminal).
        # The valid mask indicates for each context whether it actually exists or it is just a padding.
        path_source_token_input = Input((self.config.MAX_CONTEXTS,), dtype=tf.int32)
        path_input = Input((self.config.MAX_CONTEXTS,), dtype=tf.int32)
        path_target_token_input = Input((self.config.MAX_CONTEXTS,), dtype=tf.int32)
        context_valid_mask = Input((self.config.MAX_CONTEXTS,))

        inputs = [path_source_token_input, path_input, path_target_token_input, context_valid_mask]
        code_vectors, attention_weights, target_index = self.call(inputs)
        # Wrap the layers into a Keras model, using our subtoken-metrics and the CE loss.
        self.keras_train_model = keras.Model(inputs=inputs, outputs=target_index)

        # Actual target word predictions (as strings). Used as a second output layer.
        # Used for predict() and for the evaluation metrics calculations.
        topk_predicted_words, topk_predicted_words_scores = self.topk(target_index)

        # We use another dedicated Keras model for evaluation.
        # The evaluation model outputs the `topk_predicted_words` as a 2nd output.
        # The separation between train and eval models is for efficiency.
        self.keras_eval_model = keras.Model(
            inputs=inputs, outputs=[target_index, topk_predicted_words], name="code2vec-keras-model")

        # We use another dedicated Keras function to produce predictions.
        # It have additional outputs than the original model.
        # It is based on the trained layers of the original model and uses their weights.
        predict_outputs = tuple(KerasPredictionModelOutput(
            target_index=target_index, code_vectors=code_vectors, attention_weights=attention_weights,
            topk_predicted_words=topk_predicted_words, topk_predicted_words_scores=topk_predicted_words_scores))
        self.keras_model_predict_function = K.function(inputs=inputs, outputs=predict_outputs)

    ...