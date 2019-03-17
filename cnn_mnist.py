"""Cnn MNIST"""
import os
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/tmp/mnist_cnn",
        help="Directory where model summaries and checkpoints are stored",
    )
    parser.add_argument(
        "--learning-rate", default=0.001, type=float, help="The learning rate"
    )
    parser.add_argument(
        "--dropout-rate", default=0.15, type=float, help="The dropout rate"
    )
    parser.add_argument(
        "--train-batch-size",
        default=128,
        type=int,
        help="Batch Size when training",
    )
    parser.add_argument(
        "--epochs", default=20, type=int, help="Batch Size when training"
    )
    args = parser.parse_args()

    ds, info = tfds.load("mnist", with_info=True, as_supervised=True)
    ds_train, ds_test = ds["train"], ds["test"]

    def convert_types(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    ds_train = (
        ds_train.map(convert_types)
        .shuffle(1000)
        .batch(args.train_batch_size)
        .prefetch(AUTOTUNE)
    )
    ds_test = ds_test.map(convert_types).batch(32)

    model = models.Sequential(
        [
            layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(28, 28, 1)
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(args.dropout_rate),
            layers.Dense(10, activation="softmax"),
        ]
    )

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name="train_accuracy"
    )

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name="test_accuracy"
    )

    log_dir = os.path.join(args.model_dir, "logs")
    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, "train")
    )
    test_summary_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, "test")
    )

    @tf.function
    def train_step(image, label):
        with tf.GradientTape() as tape:
            predictions = model(image)
            loss = loss_object(label, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(label, predictions)

    @tf.function
    def test_step(image, label):
        predictions = model(image)
        t_loss = loss_object(label, predictions)

        test_loss(t_loss)
        test_accuracy(label, predictions)

    for epoch in range(args.epochs):
        for image, label in ds_train:
            train_step(image, label)
        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), step=epoch)

        for test_image, test_label in ds_test:
            test_step(test_image, test_label)
        with test_summary_writer.as_default():
            tf.summary.scalar("loss", test_loss.result(), step=epoch)
            tf.summary.scalar("accuracy", test_accuracy.result(), step=epoch)

        template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"
        print(
            template.format(
                epoch + 1,
                train_loss.result(),
                train_accuracy.result() * 100,
                test_loss.result(),
                test_accuracy.result() * 100,
            )
        )

        # Reset metrics every epoch
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()
