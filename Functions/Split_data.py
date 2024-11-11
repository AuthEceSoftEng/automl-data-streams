def split_data(dictionary: dict, target_key: str):
    """
        splits each data sample into the features and the target

        Args:
            dictionary: a dictionary that contains the sample with te real value
            target_key: the name of the feature that we want to predict

        returns:
            features: A dictionary contains the features of the sample
            target: The real value of the target variable of the sample


    """

    features = {key: value for key, value in dictionary.items() if key != target_key}

    target = dictionary.get(target_key)

    return features, target
