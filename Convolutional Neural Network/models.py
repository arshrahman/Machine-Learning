import tensorflow as tf
import utils

def conv_model_with_layers_api(input_layer, dropout_rate, mode):
    """
    Builds a model by using tf.layers API.

    Note that in mnist_fc_with_summaries.ipynb weights and biases are
    defined manually. tf.layers API follows similar steps in the background.
    (you can check the difference between tf.nn.conv2d and tf.layers.conv2d)
    """
    with tf.name_scope("network"):
        # Convolutional Layer #1
        # Computes 32 features using a 5x5 filter with ReLU activation.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 28, 28, 1]
        # Output Tensor Shape: [batch_size, 28, 28, 32]
        with tf.name_scope("cnn1"):
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=utils.CNN_FILTER1,
                kernel_size=[utils.KERNEL_SIZE, utils.KERNEL_SIZE],
                padding="same",
                activation=tf.nn.relu)

        # Pooling Layer #1
        # First max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 28, 28, 32]
        # Output Tensor Shape: [batch_size, 14, 14, 32]
        with tf.name_scope("pooling1"):
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[utils.POOL_SIZE, utils.POOL_SIZE], strides=utils.STRIDES)

        # Convolutional Layer #2
        # Computes 64 features using a 5x5 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 14, 14, 32]
        # Output Tensor Shape: [batch_size, 14, 14, 64]
        with tf.name_scope("cnn2"):
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=utils.CNN_FILTER2,
                kernel_size=[utils.KERNEL_SIZE, utils.KERNEL_SIZE],
                padding="same",
                activation=tf.nn.relu)

        # Pooling Layer #2
        # Second max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 14, 14, 64]
        # Output Tensor Shape: [batch_size, 7, 7, 64]
        with tf.name_scope("pooling2"):
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[utils.POOL_SIZE, utils.POOL_SIZE], strides=utils.STRIDES)

        
        
        # cnn3
        with tf.name_scope("cnn3"):
            conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=utils.CNN_FILTER3,
                kernel_size=[utils.KERNEL_SIZE, utils.KERNEL_SIZE],
                padding="same",
                activation=tf.nn.relu)

        with tf.name_scope("pooling3"):
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[utils.POOL_SIZE, utils.POOL_SIZE], strides=utils.STRIDES)

        
        # cnn4
        with tf.name_scope("cnn4"):
            conv4 = tf.layers.conv2d(
                inputs=pool3,
                filters=utils.CNN_FILTER4,
                kernel_size=[utils.KERNEL_SIZE, utils.KERNEL_SIZE],
                padding="same",
                activation=tf.nn.relu)

        with tf.name_scope("pooling4"):
            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[utils.POOL_SIZE, utils.POOL_SIZE], strides=utils.STRIDES)

        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 7, 7, 64]
        # Output Tensor Shape: [batch_size, 7 * 7 * 64]
        print(pool4.shape)
        with tf.name_scope("flatten"):
            #pool2_flat = tf.reshape(pool2, [-1, pool2.shape[1] * pool2.shape[2] * pool2.shape[3]])
            pool4_flat = tf.reshape(pool4, [-1, 5*5*256])

        # Dense Layer
        # Densely connected layer with 1024 neurons
        # Input Tensor Shape: [batch_size, 7 * 7 * 64]
        # Output Tensor Shape: [batch_size, 1024]
        with tf.name_scope("dense"):
            dense = tf.layers.dense(inputs=pool4_flat, units=utils.DENSE_UNITS, activation=tf.nn.relu)

        # Add dropout operation
        with tf.name_scope("dropout"):
            dropout = tf.layers.dropout(inputs=dense, rate=dropout_rate, training=mode)

        # Logits layer for 10 classes.
        # Input Tensor Shape: [batch_size, 1024]
        # Output Tensor Shape: [batch_size, 10]
        with tf.name_scope("logits"):
            logits = tf.layers.dense(inputs=dense, units=utils.NUM_CHANNELS)

        return logits
