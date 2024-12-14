from AutoML_pipeline.AutoML_pipeline_class import AutomlPipeline
from Functions.Split_data import split_data


def use_automl(data, target, data_drift_detector, consept_drift_detector, seed: int | None = None):

    pipeline = AutomlPipeline(target, data_drift_detector, consept_drift_detector, seed)

    pipeline.init_train(data[:pipeline.buffer_size])

    y_real = [None]*pipeline.buffer_size
    y_pred = [None]*pipeline.buffer_size

    for i in range(len(data) - pipeline.buffer_size):

        x, y = split_data(data[i + pipeline.buffer_size], target)

        y_real.append(y)

        y_pred.append(pipeline.predict_one(x))

        pipeline.learn_one(x, y)

    return y_real, y_pred, pipeline.data_drifts, pipeline.concept_drifts

