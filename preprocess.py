import numpy as np
import pandas as pd
from config.config_parser import Config


conf = Config("config/system.config")
train_path = conf.get_config("PATHS", "train_path")
test_path = conf.get_config("PATHS", "test_path")


def read_data(file_path=None, delim=","):
    """Reads file to pandas dataframe"""
    df = pd.read_csv(file_path, sep=delim)
    return df


def generate_train_validation_data(file_path):
    """Generates train and validation data and saves to numpy file"""
    df = read_data(file_path)
    Y = pd.get_dummies(df["label"]).values
    X = df.drop("label", axis=1).values
    train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)
    validation_index = np.array(list(set(range(len(X))) - set(train_index)))
    train_X = X[train_index]
    train_Y = Y[train_index]
    validation_X = X[validation_index]
    validation_Y = Y[validation_index]
    np.save(conf.get_config("PATHS", "train_X_path"), train_X)
    np.save(conf.get_config("PATHS", "train_Y_path"), train_Y)
    np.save(conf.get_config("PATHS", "validation_X_path"), validation_X)
    np.save(conf.get_config("PATHS", "validation_Y_path"), validation_Y)


def generate_test_data(file_path):
    """Generates test data and saves to numpy file"""
    df = read_data(file_path)
    X = df.values
    np.save(conf.get_config("PATHS", "test_X_path"), X)


generate_train_validation_data(train_path)
generate_test_data(test_path)


