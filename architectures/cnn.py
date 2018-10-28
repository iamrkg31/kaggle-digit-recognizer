import tensorflow as tf

class CNN(object):
    def __init__(self, filter_sizes, num_filters, input_X_shape, num_classes=10, learning_rate=0.001):
        # Placeholders
        self.input_x = tf.placeholder(tf.float32, [None, input_X_shape[0], input_X_shape[1]])
        self.input_x_expanded = tf.expand_dims(self.input_x, -1)
        self.input_y = tf.placeholder(tf.float32, [None, num_classes])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            # Convolution Layer
            filter_shape = [filter_size, input_X_shape[1], 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
            conv = tf.nn.conv2d(
                self.input_x_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b))
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, input_X_shape[0] - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs,3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        h_tanh = tf.nn.tanh(h_pool_flat)
        h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        # Create weight and bias variable
        self.weight = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1))
        self.bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

        # Create logits
        self.logits_out = tf.matmul(h_drop, self.weight) + self.bias

        # Loss function
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_out, labels=self.input_y)
        self.loss = tf.reduce_mean(losses)

        # Optimizer
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        self.goal = optimizer.minimize(self.loss)

        # Accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits_out, 1), tf.argmax(self.input_y, 1)), tf.float32))