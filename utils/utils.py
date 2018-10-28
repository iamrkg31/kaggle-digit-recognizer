import numpy as np
import matplotlib.pyplot as plt


# def generate_batch(X, Y, batch_size, epochs, shuffle=True):
#     """Generates a batch iterator for a dataset"""
#     num_batches_per_epoch = int((len(X) - 1) / batch_size) + 1
#     for epoch in range(epochs):
#         if shuffle:
#             shuffled_indices = np.random.permutation(np.arange(len(X)))
#             shuffled_X = X[shuffled_indices]
#             shuffled_Y = Y[shuffled_indices]
#         else:
#             shuffled_X = X
#             shuffled_Y = Y
#         for batch_num in range(num_batches_per_epoch):
#             start_index = batch_num * batch_size
#             end_index = min((batch_num + 1) * batch_size, len(X))
#             yield shuffled_X[start_index:end_index], shuffled_Y[start_index:end_index]


def generate_batch(X, Y, batch_size, shuffle=True):
    """Generates a batch iterator for a dataset"""
    num_batches_per_epoch = int((len(X) - 1) / batch_size) + 1
    if shuffle:
        shuffled_indices = np.random.permutation(np.arange(len(X)))
        shuffled_X = X[shuffled_indices]
        shuffled_Y = Y[shuffled_indices]
    else:
        shuffled_X = X
        shuffled_Y = Y
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, len(X))
        yield shuffled_X[start_index:end_index], shuffled_Y[start_index:end_index]


def plot_graph(epochs, list_, list_label="Test Set", x_label="Epochs",
               y_label="Accuracy", loc_="upper left", title="Accuracy"):
    """Plots graph over test and training dataset"""
    epoch_seq = np.arange(1, epochs + 1)
    plt.plot(epoch_seq, list_, 'r-', label=list_label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc_)
    plt.show()