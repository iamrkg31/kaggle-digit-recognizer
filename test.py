import numpy as np
import pandas as pd
import tensorflow as tf
from config.config_parser import Config

# Create session and graph object
sess = tf.Session()
graph = tf.get_default_graph()

# Variables
conf = Config("config/system.config")
checkpoint_file = "checkpoints/best_validation"

# Restore tf model
saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
saver.restore(sess, checkpoint_file)

# Get the placeholders from the graph by name
input_x = graph.get_operation_by_name("input_X").outputs[0]
dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
prediction = graph.get_operation_by_name("predictions").outputs[0]

# Import test data
X_ = np.load(conf.get_config("PATHS", "test_X_path"))
X = X_.reshape(X_.shape[0], conf.get_config("PARAMS", "img_height"), conf.get_config("PARAMS", "img_width"))

# Perform predictions
test_dict = {input_x: X, dropout_keep_prob: 1.0}
pred = sess.run(prediction, feed_dict=test_dict)

# Write the predictions to file
image_ids = np.array([i+1 for i in range(len(X))])
out = pd.DataFrame({'ImageId':image_ids,'Label':pred})
out.to_csv("output/digit_recognizer_out3.csv", index=False)