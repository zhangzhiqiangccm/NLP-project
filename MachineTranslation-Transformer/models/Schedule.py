#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Author zhang

import tensorflow as tf
from tensorflow import keras

#带自定义学习率调整的Adam优化器
class CustomizedSchedule(
    keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps = 4000):
        super(CustomizedSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** (-1.5))

        arg3 = tf.math.rsqrt(self.d_model)

        return arg3 * tf.math.minimum(arg1, arg2)