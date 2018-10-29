import tensorflow as tf

class CNN(object):
    def __init__(self, filter_sizes, num_filters, input_X_shape, num_classes=10, learning_rate=0.001, channels=1):
        # Placeholders
        self.input_x = tf.placeholder(tf.float32, [None, input_X_shape[0], input_X_shape[1]], name="input_X")
        self.input_x_expanded = tf.expand_dims(self.input_x, -1)
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_Y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Create a convolution and maxpool layer for each filter size
        pooled_output = self.input_x_expanded
        in_channels = channels
        out_channels = num_filters
        for i, filter_size in enumerate(filter_sizes):
            # Convolution Layer
            filter_shape = [filter_size, filter_size, in_channels, out_channels]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[out_channels]))
            conv = tf.nn.conv2d(
                pooled_output,
                W,
                strides=[1, 1, 1, 1],
                padding="SAME",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b))
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, filter_size, filter_size, 1],
                strides=[1, filter_size, filter_size, 1],
                padding='SAME',
                name="pool")
            pooled_output = pooled
            in_channels = out_channels
            out_channels = 2 * out_channels

        # Flatten the pooled result
        h_pool_flat = tf.reshape(pooled_output, [-1, int(out_channels/2)])
        h_tanh = tf.nn.tanh(h_pool_flat)
        h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        # Create weight and bias variable
        self.weight = tf.Variable(tf.truncated_normal([int(out_channels/2), num_classes], stddev=0.1))
        self.bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

        # Create logits
        self.logits_out = tf.nn.xw_plus_b(h_drop, self.weight, self.bias, name="logits_out")

        # prediction
        self.predictions = tf.argmax(self.logits_out, 1, name="predictions")

        # Loss function
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_out, labels=self.input_y)
        self.loss = tf.reduce_mean(losses)

        # Optimizer
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        self.goal = optimizer.minimize(self.loss)

        # Accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits_out, 1), tf.argmax(self.input_y, 1)), tf.float32))