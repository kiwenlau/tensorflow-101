# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from parse_csv import parse_csv


def test(model):

    test_dataset = tf.data.TextLineDataset("/tensorflow-101/data/iris_test.csv")
    test_dataset = test_dataset.skip(1)             # skip header row
    test_dataset = test_dataset.map(parse_csv)      # parse each row with the funcition created earlier
    test_dataset = test_dataset.shuffle(1000)       # randomize
    test_dataset = test_dataset.batch(32)           # use the same batch size as the training set

    test_accuracy = tfe.metrics.Accuracy()

    for (x, y) in test_dataset:
        prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

    print "\n测试:"
    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

    return
