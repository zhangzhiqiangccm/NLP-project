#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Author zhang

import tensorflow as tf
from tensorflow import keras
from models.EncoderModel import EncoderModel
from models.DecoderModel import DecoderModel

class Transformer(keras.Model):
    def __init__(self, config, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder_model = EncoderModel(
            config.num_layers, config.input_vocab_size, config.max_length,
            config.d_model, config.num_heads, config.dff, rate)

        self.decoder_model = DecoderModel(
            config.num_layers, config.target_vocab_size, config.max_length,
            config.d_model, config.num_heads, config.dff, rate)

        self.final_layer = keras.layers.Dense(config.target_vocab_size)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inp, tar, training, encoder_padding_mask,
             decoder_mask, encoder_decoder_padding_mask):
        # encoding_outputs.shape: (batch_size, input_seq_len, d_model)
        encoding_outputs = self.encoder_model(
            inp, training, encoder_padding_mask)

        # decoding_outputs.shape: (batch_size, output_seq_len, d_model)
        decoding_outputs, attention_weights = self.decoder_model(
            tar, encoding_outputs, training,
            decoder_mask, encoder_decoder_padding_mask)

        # predictions.shape: (batch_size, output_seq_len, target_vocab_size)
        predictions = self.final_layer(decoding_outputs)

        return predictions, attention_weights