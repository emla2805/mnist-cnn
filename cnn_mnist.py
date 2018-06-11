"""Cnn MNIST"""

import os
import tensorflow as tf

from argparse import ArgumentParser

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode, params):
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=6,
        kernel_size=[6, 6],
        strides=1,
        padding='SAME',
        activation=tf.nn.relu)

    drop1 = tf.layers.dropout(
        inputs=conv1, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=12,
        kernel_size=[5, 5],
        strides=2,
        padding='SAME',
        activation=tf.nn.relu)

    drop2 = tf.layers.dropout(
        inputs=conv2, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    conv3 = tf.layers.conv2d(
        inputs=drop2,
        filters=24,
        kernel_size=[4, 4],
        strides=2,
        padding='SAME',
        activation=tf.nn.relu)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 24]
    # Output Tensor Shape: [batch_size, 7 * 7 * 24]
    conv3_flat = tf.reshape(conv3, [-1, 7 * 7 * 24])

    # Dense Layer
    # Densely connected layer with 200 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 24]
    # Output Tensor Shape: [batch_size, 200]
    dense = tf.layers.dense(inputs=conv3_flat, units=200, activation=tf.nn.relu)

    # Add dropout operation; 0.25 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 200]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)

    tf.summary.histogram('conv1', conv1)
    tf.summary.histogram('conv2', conv2)
    tf.summary.histogram('conv3', conv3)
    tf.summary.histogram('dense', dense)
    tf.summary.histogram('logits', logits)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        'classes': tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predictions['classes'])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy)  # Add accuracy to TRAIN also

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def data_input_fn(file_pattern, batch_size=1000, shuffle=False):
    def _parser(record):
        keys_to_features = {
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        label = tf.cast(parsed.pop('label'), dtype=tf.int32)

        image = tf.decode_raw(parsed['image_raw'], tf.uint8)
        image.set_shape((28 * 28))

        # Normalize image
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

        return image, label

    files = tf.data.Dataset.list_files(file_pattern)
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=32)

    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(_parser, batch_size=batch_size, num_parallel_batches=4))

    if shuffle:
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=10_000, count=None))
    else:
        dataset = dataset.repeat(count=1)

    dataset = dataset.prefetch(2)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    return images, labels


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--data-directory',
        default='/tmp/data/mnist',
        help='Directory where the data is stored'
    )
    parser.add_argument(
        '--model-directory',
        default='/tmp/mnist_cnn',
        help='Directory where model summaries and checkpoints are stored'
    )
    parser.add_argument(
        '--learning-rate',
        default=0.003,
        type=float,
        help='The learning rate'
    )
    parser.add_argument(
        '--dropout-rate',
        default=0.75,
        type=float,
        help='The dropout rate'
    )
    parser.add_argument(
        '--train-batch-size',
        default=128,
        type=int,
        help='Batch Size when training'
    )
    args = parser.parse_args()

    run_config = tf.estimator.RunConfig().replace(
        save_summary_steps=600,
        save_checkpoints_secs=600,
    )

    # Create the Estimator
    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        config=run_config,
        model_dir=args.model_directory,
        params={
            'learning_rate': args.learning_rate,
            'dropout_rate': args.dropout_rate
        }
    )

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    def train_input_fn():
        return data_input_fn(
            os.path.join(args.data_directory, 'train.tfrecords'),
            batch_size=args.train_batch_size,
            shuffle=True
        )

    def eval_input_fn():
        return data_input_fn(
            os.path.join(args.data_directory, 'test.tfrecords'),
            batch_size=1000
        )

    # feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    # serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=40000)
    # export_latest = tf.estimator.LatestExporter(
    #     name='serving', serving_input_receiver_fn=serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn, throttle_secs=30)#, exporters=export_latest, steps=None)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

