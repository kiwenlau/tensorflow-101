# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from parse_csv import parse_csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.enable_eager_execution()


# 导入训练数据
TRAIN_DATASET = tf.data.TextLineDataset("/tensorflow-101/data/iris_training.csv")
TRAIN_DATASET = TRAIN_DATASET.skip(1)             # skip the first header row
TRAIN_DATASET = TRAIN_DATASET.map(parse_csv)      # parse each row
TRAIN_DATASET = TRAIN_DATASET.shuffle(buffer_size=1000)  # randomize
TRAIN_DATASET = TRAIN_DATASET.batch(32)


# 定义神经网络
MODEL = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),  # input shape required
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(3)
])


# 损失计算函数
def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


# 梯度计算函数
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, MODEL.variables)


# 优化器
OPTIMIZER = tf.train.GradientDescentOptimizer(learning_rate=0.01)


def train():

    print("训练:")

    num_epochs = 201

    for epoch in range(num_epochs):

        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()

        for x, y in TRAIN_DATASET:

            # 计算梯度
            grads = grad(MODEL, x, y)

            # 优化模型的参数
            OPTIMIZER.apply_gradients(zip(grads, MODEL.variables), global_step=tf.train.get_or_create_global_step())

            # 计算损失
            epoch_loss_avg(loss(MODEL, x, y))

            # 计算准确度
            epoch_accuracy(tf.argmax(MODEL(x), axis=1, output_type=tf.int32), y)

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))

    return MODEL
  