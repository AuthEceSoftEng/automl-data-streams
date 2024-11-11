# script that reads a csv and convert it to a list of dict
import pandas as pd

# dataset is the dataset from river or the path of CSV and


def prepare_data(dataset):
    """
        Prepares the dataset for the model

        Args:
            dataset: The dataset we used (path of a csv or a river library)

        returns:
            data: dataset in list of dictionaries

    """

    if isinstance(dataset, str):
        # open the csv with data
        data = pd.read_csv(dataset)
        # convert them to an array of dicts
        data = data.to_dict(orient='records')

    else:
        # prepare the dataset for the model
        data = []
        for i, (x, y) in enumerate(dataset):
            data.append({**x, "y": y})

    return data
