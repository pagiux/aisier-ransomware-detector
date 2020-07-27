import pandas as pd


def prepare_dataset(filename, label):
    # simply read as csv
    data = pd.read_csv(filename, sep=',', header=None)
    data[len(data.columns)] = label

    return data