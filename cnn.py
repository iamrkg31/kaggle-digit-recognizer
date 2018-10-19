import numpy as np
import pandas as pd

# Import data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/train.csv")
train_Y = pd.get_dummies(train_data["label"], dtype=np.int32).values
train_X = train_data.drop("label",axis=1).values
test_Y = pd.get_dummies(test_data["label"], dtype=np.int32).values
test_X = test_data.drop("label",axis=1).values

print(train_X.shape)