from river import preprocessing, feature_selection, stats
import copy
import random
from river import compose
from river import linear_model, naive_bayes, neighbors, tree, metrics, forest
from concurrent.futures import ProcessPoolExecutor


def pipeline_test(model, preprocessor, feature_selector, x_train, y_train):

    accuracy = metrics.Accuracy()
    pipeline = None
    try:
        if preprocessor is not None and feature_selector is not None:
            pipeline = compose.Pipeline(copy.deepcopy(preprocessor),
                                        copy.deepcopy(feature_selector),
                                        copy.deepcopy(model))

        elif preprocessor is None and feature_selector is None:
            pipeline = copy.deepcopy(model)

        elif preprocessor is None and feature_selector is not None:
            pipeline = compose.Pipeline(copy.deepcopy(feature_selector), copy.deepcopy(model))

        elif preprocessor is not None and feature_selector is None:
            pipeline = compose.Pipeline(copy.deepcopy(preprocessor), copy.deepcopy(model))

        for i in range(len(x_train)):
            x = x_train[i]

            if i > 5:
                y_pred = pipeline.predict_one(x)
                accuracy.update(y_train[i], y_pred)

            pipeline.learn_one(x, y_train[i])

        return [pipeline, accuracy]

    except Exception as e:
        # print(f"Error message: {str(e)}")
        return None


def find_best_pipeline(x_train, y_train, data_drift_detector_method, concept_drift_detector_method, seed: int | None = None):
    random.seed(seed)
    preprocessors = [None, preprocessing.StandardScaler(),
                     preprocessing.MinMaxScaler()]  # preprocessing.AdaptiveStandardScaler(), preprocessing.Normalizer()

    feature_selectors = [None, feature_selection.VarianceThreshold(), feature_selection.SelectKBest(
        similarity=stats.PearsonCorr(),
        k=3)]  # feature_selection.SelectKBest(similarity=stats.PearsonCorr(), k=4), feature_selection.SelectKBest(similarity=stats.PearsonCorr(), k=2)

    concept_drift_detector = copy.deepcopy(concept_drift_detector_method)

    # take the features of the dataset
    features = [key for key in x_train[0].keys()]

    data_drift_detectors = None

    # create data drift detectors for every feature
    if data_drift_detector_method is not None:
        data_drift_detectors = {feature: copy.deepcopy(data_drift_detector_method) for feature in features}

    ALMAClassifier = [linear_model.ALMAClassifier(p=p, alpha=alpha) for p in [1, 2, 3] for alpha in
                      [0.9, 0.8]]

    LogisticRegression = [linear_model.LogisticRegression(l2=l2, intercept_init=intercept_init)
                          for l2 in [0, 0.01, 0.05, 0.1] for intercept_init in
                          [0, 0.1]]

    PAClassifier = [linear_model.PAClassifier(C=C, mode=mode) for C in [0.01, 0.5, 1.0]
                    for mode in [1, 2]]

    Perceptron = [linear_model.Perceptron(l2=l2) for l2 in [0, 0.01, 0.05]]

    GaussianNB = [naive_bayes.GaussianNB()]

    KNNClassifier = [neighbors.KNNClassifier(n_neighbors=n_neighbors)
                     for n_neighbors in [2, 3, 4, 5, 6, 7, 8]]

    ExtremelyFastDecisionTreeClassifier = [tree.ExtremelyFastDecisionTreeClassifier(grace_period=grace_period,
                                                                                    delta=delta)
                                           for grace_period in [50, 100, 200] for delta in
                                           [1e-07, 1e-06, 1e-05]]

    HoeffdingTreeClassifier = [tree.HoeffdingTreeClassifier(grace_period=grace_period,
                                                            delta=delta)
                               for grace_period in [50, 100, 200] for delta in
                               [1e-07, 1e-06, 1e-05]]

    SGTClassifier = [tree.SGTClassifier(grace_period=grace_period, delta=delta)
                     for grace_period in [50, 100, 200] for delta in
                     [1e-07, 1e-06, 1e-05]]

    HoeffdingAdaptiveTreeClassifier = [tree.HoeffdingAdaptiveTreeClassifier(grace_period=grace_period,
                                                                            delta=delta, seed = seed)
                                       for grace_period in [50, 100, 200] for delta in
                                       [1e-07, 1e-06, 1e-05]]  

    AMFClassifier = [forest.AMFClassifier(n_estimators=n_estimators, seed = seed) for n_estimators in [5, 10, 15]]

    ARFClassifier = [forest.ARFClassifier(n_models=n_models, seed = seed) for n_models in [5, 10, 15]]

    models = []

    for algorithms in [ALMAClassifier, LogisticRegression, PAClassifier, Perceptron, GaussianNB, KNNClassifier,
                       ExtremelyFastDecisionTreeClassifier, HoeffdingTreeClassifier, SGTClassifier,
                       HoeffdingAdaptiveTreeClassifier, AMFClassifier, ARFClassifier]:

        models.extend(algorithms)

    pipeline_can_be_used = []

    best_pipeline = None

    with ProcessPoolExecutor() as executor:

        results = []
        for model in models:
            for preprocessor in preprocessors:
                for feature_selector in feature_selectors:
                    results.append(executor.submit(pipeline_test, model, preprocessor, feature_selector, x_train, y_train))

        for result in results:
            temp_result = result.result()
            if temp_result is not None:
                pipeline_can_be_used.append(temp_result)

    # Extract all accuracies (last column)
    accuracy_values = [pipeline[-1].get() for pipeline in pipeline_can_be_used]

    # find the best accuracy
    best_accuracy = max(accuracy_values)

    best_accuracy_indexes = []

    # find the indexes of the best models
    for i in range(len(accuracy_values)):

        if accuracy_values[i] == best_accuracy:
            best_accuracy_indexes.append(i)

    if len(best_accuracy_indexes) == 1:
        best_pipeline = pipeline_can_be_used[best_accuracy_indexes[0]]

    elif len(best_accuracy_indexes) > 1:
        choose_one = random.randint(0, len(best_accuracy_indexes) - 1)
        best_pipeline = pipeline_can_be_used[best_accuracy_indexes[choose_one]]

    else:
        print("Error, no model can be used")

    return best_pipeline[0], best_pipeline[1], data_drift_detectors, concept_drift_detector
