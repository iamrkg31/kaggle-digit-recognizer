import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Start a graph
sess = tf.Session()

img_height = 28
img_width = 28
channels = 1
filter_sizes = [2,3,4]
num_filters = 128
epochs = 100
batch_size = 250
max_sequence_length = 25
rnn_size = 15
embedding_size = 50
min_word_frequency = 10
learning_rate = 0.0001
num_classes = 10

# Import data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
Y = pd.get_dummies(train_data["label"], dtype=np.int32).values
X = train_data.drop("label",axis=1).values
# test_Y = pd.get_dummies(test_data["label"], dtype=np.int32).values
# test_X = test_data.drop("label",axis=1).values

# Divide the inputs into train and test data
# 80% training data, 20% testing data
train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)
test_index = np.array(list(set(range(len(X))) - set(train_index)))
train_X = X[train_index]
train_Y = Y[train_index]
test_X = X[test_index]
test_Y = Y[test_index]

# Create placeholders
input = tf.placeholder(tf.float32, [None, img_height, img_width, channels])
target = tf.placeholder(tf.int32, [None,train_Y.shape[1]])
dropout_keep_prob = tf.placeholder(tf.float32)

# CNN architecture
pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    # Convolution Layer
    filter_shape = [filter_size, filter_size, channels, num_filters]
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
    conv = tf.nn.conv2d(
        input,
        W,
        strides=[1, 1, 1, 1],
        padding="SAME",
        name="conv")
    # Apply nonlinearity
    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
    # Max-pooling over the outputs
    pooled = tf.nn.max_pool(
        h,
        ksize=[1, filter_size, filter_size, 1],
        strides=[1, 1, 1, 1],
        padding='SAME',
        name="pool")
    pooled_outputs.append(pooled)

# Combine all the pooled features
num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(pooled_outputs,3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total]) # -1 means "all"
h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

# Create weight and bias variable
weight = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

# Create logits
logits_out = tf.matmul(h_drop, weight) + bias

# Loss function
losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits_out, labels=target)
loss = tf.reduce_mean(losses)

# Accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.argmax(target, 1)), tf.float32))

# Optimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate)
goal = optimizer.minimize(loss)

# Initialize variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Define the variable that stores the result
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

# Start training
for epoch in range(epochs):
    # Shuffle training data
    shuffled_ix = np.random.permutation(np.arange(len(train_X)))
    train_X = train_X[shuffled_ix]
    train_Y = train_Y[shuffled_ix]
    num_batches = int(len(train_X) / batch_size) + 1

    for i in range(num_batches):
        # Select train data
        batch_index = np.random.choice(len(train_X), size=batch_size)
        batch_train_X = train_X[batch_index]
        batch_train_X = batch_train_X.reshape(batch_size,img_height,img_width,channels)
        batch_train_y = train_Y[batch_index]
        # Run train step
        train_dict = {input: batch_train_X, target: batch_train_y, dropout_keep_prob: 0.5}
        sess.run(goal, feed_dict=train_dict)

    # Run loss and accuracy for training
    temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
    train_loss.append(temp_train_loss)
    train_accuracy.append(temp_train_acc)

    # Run Eval Step
    test_dict = {input: test_X, target: test_Y, dropout_keep_prob: 1.0}
    temp_test_loss, temp_test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
    test_loss.append(temp_test_loss)
    test_accuracy.append(temp_test_acc)
    print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch + 1, temp_test_loss, temp_test_acc))

print('\nOverall accuracy on test set (%): {}'.format(np.mean(temp_test_acc) * 100.0))

# Plot loss over time
epoch_seq = np.arange(1, epochs + 1)
plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
plt.plot(epoch_seq, test_loss, 'r-', label='Test Set')
plt.title('training/test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.show()

# Plot accuracy over time
plt.plot(epoch_seq, train_accuracy, 'k--', label='Train Set')
plt.plot(epoch_seq, test_accuracy, 'r-', label='Test Set')
plt.title('Test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()