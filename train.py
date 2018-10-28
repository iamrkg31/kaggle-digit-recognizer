import numpy as np
import tensorflow as tf
from architectures.cnn import CNN
from config.config_parser import Config
from utils import utils

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
evaluate_every = conf.get_config("PARAMS", "evaluate_every")

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

for epoch in range(epochs):
    batch_iter = utils.generate_batch(train_X, train_Y, batch_size, shuffle=True)
    for batch_X, batch_Y in batch_iter:
        train_dict = {cnn.input_x: batch_X, cnn.input_y: batch_Y, cnn.dropout_keep_prob: 0.5}
        sess.run(cnn.goal, feed_dict=train_dict)
    # Train loss and accuracy
    temp_train_loss, temp_train_acc = sess.run([cnn.loss, cnn.accuracy], feed_dict=train_dict)
    train_loss.append(temp_train_loss)
    train_accuracy.append(temp_train_acc)
    print('Epoch: {}, Train Loss: {:.2}, Train Acc: {:.2}'.format(epoch + 1, temp_train_loss, temp_train_acc))
    # Test loss and accuracy
    if (epoch+1) % evaluate_every == 0:
        print("Evaluation*********************")
        test_dict = {cnn.input_x: validation_X, cnn.input_y: validation_Y, cnn.dropout_keep_prob: 1.0}
        temp_test_loss, temp_test_acc = sess.run([cnn.loss, cnn.accuracy], feed_dict=test_dict)
        test_loss.append(temp_test_loss)
        test_accuracy.append(temp_test_acc)
        print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch + 1, temp_test_loss, temp_test_acc))
        print("*******************************")

# Plot loss over time
utils.plot_graph(len(train_loss), train_loss, list_label="Train", x_label="Epochs", y_label="Loss", title="Loss")
utils.plot_graph(len(test_loss), test_loss, list_label="Test", x_label="Epochs", y_label="Loss", title="Loss")
# Plot accuracy over time
utils.plot_graph(len(train_accuracy), train_accuracy, list_label="Train", x_label="Epochs", y_label="Accuracy", title="Accuracy")
utils.plot_graph(len(test_accuracy), test_accuracy, list_label="Test", x_label="Epochs", y_label="Accuracy", title="Accuracy")
