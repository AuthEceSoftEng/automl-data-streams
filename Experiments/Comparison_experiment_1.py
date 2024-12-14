from river import metrics, preprocessing, tree, forest
from Functions.Result_extractor import result_extractor
from LoanDataset.Create_loandataset import create_loandataset
from Functions.Create_Plots import create_plots
from Functions.Data_plot import data_plot
from Simple_pipeline.Simple_pipeline_use import simple_pipeline
from Functions.Evaluation_sliding_window import evaluation
from river import linear_model
from AutoML_pipeline.AutoML_Usage import use_automl
import time


"""

    An example of how to use all the functions of the project

"""
if __name__ == "__main__":
    seed = 30
    # create dataset
    data = create_loandataset(2, datalimit=20000,
                              conceptdriftpoints={2500: "growth", 4000: "crisis",
                                                  6000: "growth", 10000: "normal",
                                                  13000: "growth", 16000: "normal",
                                                  18000: "crisis"},
                              datadriftpoints={2000: "crisis", 6000: "growth",
                                               8000: "normal", 9000: "crisis",
                                               12000: "growth", 14000: "crisis",
                                               16000: "normal"},
                              seed=seed)

    # set the target name
    target = 'y'

    # create the lists to save the results
    y_real = []
    y_predicted = []
    data_drifts = []
    concept_drifts = []

    # pipeline 1
    start_time = time.time()
    y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = (
        simple_pipeline(tree.HoeffdingAdaptiveTreeClassifier(seed = seed), None, None, data, target))

    finish_time = time.time()
    total_time = finish_time - start_time
    print("Time of pipeline 1: ", total_time)

    # add the results from the pipeline in the lists
    y_real.append(y_real_temp)
    y_predicted.append(y_predicted_temp)
    data_drifts.append(data_drifts_temp)
    concept_drifts.append(concept_drifts_temp)

    result_extractor(1, 1, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

    print("-----------------------------------------------------------------------------------------------------------")
    # pipeline 2
    start_time = time.time()
    y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = (
        simple_pipeline(tree.HoeffdingAdaptiveTreeClassifier(seed = seed), preprocessing.StandardScaler, None, data, target))

    finish_time = time.time()
    total_time = finish_time - start_time
    print("Time of pipeline 2: ", total_time)

    # add the results from the pipeline in the lists
    y_real.append(y_real_temp)
    y_predicted.append(y_predicted_temp)
    data_drifts.append(data_drifts_temp)
    concept_drifts.append(concept_drifts_temp)

    result_extractor(1, 2, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

    print("-----------------------------------------------------------------------------------------------------------")
    # pipeline 3
    start_time = time.time()
    y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = (
        simple_pipeline(tree.HoeffdingTreeClassifier(), None, None, data, target))

    finish_time = time.time()
    total_time = finish_time - start_time
    print("Time of pipeline 3: ", total_time)

    # add the results from the pipeline in the lists
    y_real.append(y_real_temp)
    y_predicted.append(y_predicted_temp)
    data_drifts.append(data_drifts_temp)
    concept_drifts.append(concept_drifts_temp)

    result_extractor(1, 3, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

    print("-----------------------------------------------------------------------------------------------------------")
    # pipeline 4
    start_time = time.time()
    y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = (
        simple_pipeline(tree.HoeffdingTreeClassifier(), preprocessing.StandardScaler, None, data, target))

    finish_time = time.time()
    total_time = finish_time - start_time
    print("Time of pipeline 4: ", total_time)

    # add the results from the pipeline in the lists
    y_real.append(y_real_temp)
    y_predicted.append(y_predicted_temp)
    data_drifts.append(data_drifts_temp)
    concept_drifts.append(concept_drifts_temp)

    result_extractor(1, 4, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

    print("-----------------------------------------------------------------------------------------------------------")
    # pipeline 5
    start_time = time.time()
    y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = (
        simple_pipeline(linear_model.LogisticRegression(), None, None, data, target))

    finish_time = time.time()
    total_time = finish_time - start_time
    print("Time of pipeline 5: ", total_time)

    # add the results from the pipeline in the lists
    y_real.append(y_real_temp)
    y_predicted.append(y_predicted_temp)
    data_drifts.append(data_drifts_temp)
    concept_drifts.append(concept_drifts_temp)

    result_extractor(1, 5, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

    print("-----------------------------------------------------------------------------------------------------------")
    # pipeline 6
    start_time = time.time()
    y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = (
        simple_pipeline(linear_model.LogisticRegression(), preprocessing.StandardScaler, None, data, target))

    finish_time = time.time()
    total_time = finish_time - start_time
    print("Time of pipeline 6: ", total_time)

    # add the results from the pipeline in the lists
    y_real.append(y_real_temp)
    y_predicted.append(y_predicted_temp)
    data_drifts.append(data_drifts_temp)
    concept_drifts.append(concept_drifts_temp)

    result_extractor(1, 6, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

    print("-----------------------------------------------------------------------------------------------------------")
    # pipeline 7
    start_time = time.time()
    y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = (
        simple_pipeline(forest.AMFClassifier(seed = seed), None, None, data, target))

    finish_time = time.time()
    total_time = finish_time - start_time
    print("Time of pipeline 7: ", total_time)

    # add the results from the pipeline in the lists
    y_real.append(y_real_temp)
    y_predicted.append(y_predicted_temp)
    data_drifts.append(data_drifts_temp)
    concept_drifts.append(concept_drifts_temp)

    result_extractor(1, 7, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

    print("-----------------------------------------------------------------------------------------------------------")
    # pipeline 8
    start_time = time.time()
    y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = (
        simple_pipeline(forest.AMFClassifier(seed = seed), preprocessing.StandardScaler, None, data, target))

    finish_time = time.time()
    total_time = finish_time - start_time
    print("Time of the pipeline 8: ", total_time)

    # add the results from the pipeline in the lists
    y_real.append(y_real_temp)
    y_predicted.append(y_predicted_temp)
    data_drifts.append(data_drifts_temp)
    concept_drifts.append(concept_drifts_temp)

    result_extractor(1, 8, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

    print("-----------------------------------------------------------------------------------------------------------")
    # pipeline 9
    start_time = time.time()
    y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = (
        use_automl(data, target, True, True, seed = seed))

    finish_time = time.time()
    total_time = finish_time - start_time
    print("Time of pipeline 9: ", total_time)

    # add the results from the pipeline in the lists
    y_real.append(y_real_temp)
    y_predicted.append(y_predicted_temp)
    data_drifts.append(data_drifts_temp)
    concept_drifts.append(concept_drifts_temp)

    result_extractor(1, 9, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

    # Delete the temporal variables
    del y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp

    # evaluation of pipelines
    evaluates = evaluation(y_real, y_predicted, metrics.Accuracy())

    # plots for metrics and drifts of the pipelines
    create_plots(evaluates, data_drifts, concept_drifts)

    # plots for data
    data_plot(data, 50)


