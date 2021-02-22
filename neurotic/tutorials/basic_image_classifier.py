"""
https://www.tensorflow.org/tutorials/quickstart/beginner
"""

import tensorflow as tf

# load and prepare dataset. convert ints to floats
mnist = tf.keras.datasets.mnist
mnist_data = mnist.load_data()

x_train, y_train = mnist_data[0]
x_test,  y_test  = mnist_data[1]

x_train = x_train.astype(float) / 255.0
x_test  = x_test.astype(float) / 255.0

# build model by stacking layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
])

# prepare the model for training
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# train it
model.fit(x_train, y_train, epochs=5)