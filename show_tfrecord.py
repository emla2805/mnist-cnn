import tensorflow as tf


for example in tf.python_io.tf_record_iterator("/tmp/data/mnist/train.tfrecords"):
    result = tf.train.Example.FromString(example)
    print(result)
    break
