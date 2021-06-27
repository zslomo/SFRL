# iris_tf.py
# Iris example using TensorFlow 1.7.0

import numpy as np
import pandas as pd
import tensorflow as tf
import os


def get_iris_label(label):
    if label == "Iris-setosa":
        return [1, 0, 0]
    elif label == "Iris-versicolor":
        return [0, 1, 0]
    else:
        return [0, 0, 1]


def main():
    np.random.seed(1024)

    print("\nLoading iris train and test data \n")
    train_file = "../data/iris/iris.data"
    df = pd.read_csv(train_file, header=None)
    train_x = []
    train_y = []
    for row in df.values:
        train_x.append(row[:4])
        train_y.append(get_iris_label(row[4]))

    # prepare to create model

    X = tf.placeholder(tf.float32, shape=[None, 4])
    y = tf.placeholder(tf.float32, shape=[None, 3])
    w_1 = tf.Variable(initial_value=tf.ones([4, 6], dtype=tf.float32), name="w_1")
    b_1 = tf.Variable(initial_value=0.0, name="b_1")
    o_1 = tf.add(tf.matmul(X, w_1), b_1)
    w_2 = tf.Variable(initial_value=tf.ones([6, 3], dtype=tf.float32), name="w_2")
    b_2 = tf.Variable(initial_value=0.0, name="b_2")
    o_2 = tf.add(tf.matmul(o_1, w_2), b_2)

    # set up training
    
    cee = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=o_2)
    cost = tf.reduce_mean(cee)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    w_1_g = optimizer.compute_gradients(cee, w_1)
    trainer = optimizer.minimize(cost)

    # train model
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print("Starting training")
        max_epochs = 2
        for epoch in range(max_epochs):
            sess.run(trainer, feed_dict={X: train_x,
                                         y: train_y})
            _cee = sess.run(cee, feed_dict={X: train_x,
                                              y: train_y})
            _cost = sess.run(cost, feed_dict={X: train_x,
                                              y: train_y})
            _w_1 = sess.run(w_1, feed_dict={X: train_x,
                                           y: train_y})
            _w_1_g = sess.run(w_1_g, feed_dict={X: train_x,
                                           y: train_y})
            _w_2 = sess.run(w_2, feed_dict={X: train_x,
                                           y: train_y})
            print("cee : {}".format(_cee))
            print("_w_1_g, {}:".format(epoch))
            print(_w_1_g)
            # print("w_1, {}:".format(epoch))
            # print(_w_1)


if __name__ == "__main__":
    main()
