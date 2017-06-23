import tensorflow as tf

save_file = './model.ckpt'

tf.reset_default_graph()

weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, save_file)

    print("Weights: ")
    print(sess.run(weights))

    print("Bias: ")
    print(sess.run(bias))

    saver.save(sess, save_file)
