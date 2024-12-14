from AutoML_pipeline.Find_best_pipeline_ProcessPoolExecutor import find_best_pipeline
# from AutoML_pipeline.Find_best_pipeline import find_best_pipeline


def change_pipeline(pipeline_old, x_train, y_train, data_drift_detector_method, concept_drift_detector_method,
                    buffer_accuracy, seed: int | None = None):
    """
        Train an AutoMl pipeline to use later for predictions

        Args:
            pipeline_old: the old classifier
            x_train: data with features values
            y_train: data with targets values
            data_drift_detector_method: the data drift detection method
            concept_drift_detector_method: the consept drift detection method
            buffer_accuracy: the accuracy of the old model in the buffer
            seed: Random seed for reproducibility

        returns:
            pipeline: the trained pipeline
            accuracy: the accuracy of the pipeline
            data_drift_detectors: the data drift detectors for used features
            concept_drift_detector: the consept drift detector
        """

    # find the best pipeline
    pipeline_new, accuracy_new, data_drift_detectors_new, concept_drift_detector_new = (
        find_best_pipeline(x_train, y_train, data_drift_detector_method, concept_drift_detector_method, seed))

    # create the instance of metric
    accuracy_old = buffer_accuracy

    # if the new pipeline is better than the old use the new
    if accuracy_new.get() > accuracy_old.get():

        print("New pipeline selected: \n", pipeline_new)

        (pipeline, accuracy, data_drift_detectors, concept_drift_detector) = \
            (pipeline_new, accuracy_new, data_drift_detectors_new, concept_drift_detector_new)

    # if the old pipeline is better than the new use the old
    else:

        print("Keep the old pipeline: \n", pipeline_old)

        (pipeline, accuracy, data_drift_detectors, concept_drift_detector) =\
            (pipeline_old, accuracy_old, data_drift_detectors_new, concept_drift_detector_new)

    print("", accuracy)

    return pipeline, accuracy, data_drift_detectors, concept_drift_detector
