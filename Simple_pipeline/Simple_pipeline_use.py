from river.compose import Pipeline
from Functions.Split_data import split_data


def simple_pipeline(model, preprocessor, feature_selector, data, target):

    if preprocessor is not None and feature_selector is not None:
        pipeline = Pipeline(preprocessor(), feature_selector(), model)

    elif preprocessor is not None and feature_selector is None:
        pipeline = Pipeline(preprocessor(), model)

    elif preprocessor is None and feature_selector is not None:
        pipeline = Pipeline(feature_selector(), model)

    else:
        pipeline = model

    y_real = []
    y_pred = []

    for i in range(len(data)):
        x, y = split_data(data[i], target)

        y_real.append(y)

        y_pred.append(pipeline.predict_one(x))

        pipeline.learn_one(x, y)

    return y_real, y_pred, [], []
