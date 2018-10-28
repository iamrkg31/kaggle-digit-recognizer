import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from config.config_parser import Config
from architectures.cnn import CNN

# Start a graph
sess = tf.Session()

# Parameters
conf = Config("config/system.config")
img_height = conf.get_config("PARAMS", "img_height")
img_width = conf.get_config("PARAMS", "img_width")
channels = conf.get_config("PARAMS", "channels")
filter_sizes = conf.get_config("PARAMS", "filter_sizes")
num_filters = conf.get_config("PARAMS", "num_filters")
epochs = conf.get_config("PARAMS", "epochs")
batch_size = conf.get_config("PARAMS", "batch_size")
learning_rate = conf.get_config("PARAMS", "learning_rate")
num_classes = conf.get_config("PARAMS", "num_classes")


# Import data
train_X_ = np.load(conf.get_config("PATHS", "train_X_path"))
train_X = train_X_.reshape(train_X_.shape[0], img_height, img_width)
train_Y = np.load(conf.get_config("PATHS", "train_Y_path"))
validation_X_ = np.load(conf.get_config("PATHS", "validation_X_path"))
validation_X = validation_X_.reshape(validation_X_.shape[0], img_height, img_width)
validation_Y = np.load(conf.get_config("PATHS", "validation_Y_path"))

cnn = CNN(filter_sizes=filter_sizes, num_filters=num_filters, input_X_shape=(img_height, img_width),
          num_classes=num_classes, learning_rate=learning_rate)

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
        batch_train_y = train_Y[batch_index]
        # Run train step
        train_dict = {cnn.input_x: batch_train_X, cnn.input_y: batch_train_y, cnn.dropout_keep_prob: 0.5}
        sess.run(cnn.goal, feed_dict=train_dict)

    # Run loss and accuracy for training
    temp_train_loss, temp_train_acc = sess.run([cnn.loss, cnn.accuracy], feed_dict=train_dict)
    train_loss.append(temp_train_loss)
    train_accuracy.append(temp_train_acc)

    # Run Eval Step
    test_dict = {cnn.input_x: validation_X, cnn.input_y: validation_Y, cnn.dropout_keep_prob: 1.0}
    temp_test_loss, temp_test_acc = sess.run([cnn.loss, cnn.accuracy], feed_dict=test_dict)
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