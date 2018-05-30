# -*- coding: utf-8 -*-

import tensorflow as tf

def predict(model):

    class_ids = ["Iris setosa", "Iris versicolor", "Iris virginica"]

    predict_dataset = tf.convert_to_tensor([
        [5.1, 3.3, 1.7, 0.5,],
        [5.9, 3.0, 4.2, 1.5,],
        [6.9, 3.1, 5.4, 2.1]
    ])

    predictions = model(predict_dataset)

    print "\n预测结果:"
    for i, logits in enumerate(predictions):
        class_idx = tf.argmax(logits).numpy()
        name = class_ids[class_idx]
        print("Example {} prediction: {}".format(i, name))

    return
