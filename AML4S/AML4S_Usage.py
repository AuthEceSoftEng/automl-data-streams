from AML4S.AML4S_class import AML4S
from Functions.Split_data import split_data


def use_AML4S(data, target, data_drift_detector, consept_drift_detector, seed: int | None = None):

    # initialization
    pipeline = AML4S(target, data_drift_detector, consept_drift_detector, seed)

    # initial training with size of training data set = buffer size
    pipeline.init_train(data[:pipeline.buffer_size])

    # create list to save the results (fist positions are None because there aren't any predictions)
    y_real = [None]*pipeline.buffer_size
    y_pred = [None]*pipeline.buffer_size

    # data instances comes one by one
    for i in range(len(data) - pipeline.buffer_size):

        # split every data instance
        x, y = split_data(data[i + pipeline.buffer_size], target)

        # add the real value on the list
        y_real.append(y)

        # make prediction with the AML4S
        y_pred.append(pipeline.predict_one(x))

        # train the AML4S with the new instance
        pipeline.learn_one(x, y)

    return y_real, y_pred, pipeline.data_drifts, pipeline.concept_drifts

