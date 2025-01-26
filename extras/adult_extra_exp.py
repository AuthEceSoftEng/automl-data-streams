import numpy as np
import pandas as pd
from river import metrics, preprocessing, tree, forest, linear_model
from Functions.Data_plot import data_plot
from Functions.Evaluation import evaluation
from AML4S.AML4S_Usage import use_AML4S
from Simple_pipeline.Simple_pipeline_use import simple_pipeline
import time
from Functions.Result_extractor import result_extractor
from Functions.Create_Plots import create_plots

if __name__ == "__main__":
    seed = 30

    experiment_no = "adult"

    # Load dataset
    # open the csv with data
    df = pd.read_csv('../More_datasets/adult.data',delimiter=",")

    # set missing values to nan
    df.replace(' ?', np.nan, inplace=True)

    # encode data
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].astype('category').cat.codes.replace(-1, np.nan)

    # convert data to an array of dicts
    data = df.to_dict(orient='records')

    # set the target name
    target = '15'

    # create the lists to save the results
    y_real = []
    y_predicted = []
    data_drifts = []
    concept_drifts = []

    # pipeline 1
    start_time = time.time()
    y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = (
        simple_pipeline(tree.HoeffdingAdaptiveTreeClassifier(seed=seed), None, None, data, target))

    finish_time = time.time()
    total_time = finish_time - start_time
    print("Time of pipeline 1: ", total_time)

    # add the results from the pipeline in the lists
    y_real.append(y_real_temp)
    y_predicted.append(y_predicted_temp)
    data_drifts.append(data_drifts_temp)
    concept_drifts.append(concept_drifts_temp)

    result_extractor(experiment_no, 1, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

    print("-----------------------------------------------------------------------------------------------------------")
    # pipeline 2
    start_time = time.time()
    y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = (
        simple_pipeline(tree.HoeffdingAdaptiveTreeClassifier(seed=seed), preprocessing.StandardScaler, None, data, target))

    finish_time = time.time()
    total_time = finish_time - start_time
    print("Time of pipeline 2: ", total_time)

    # add the results from the pipeline in the lists
    y_real.append(y_real_temp)
    y_predicted.append(y_predicted_temp)
    data_drifts.append(data_drifts_temp)
    concept_drifts.append(concept_drifts_temp)

    result_extractor(experiment_no, 2, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

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

    result_extractor(experiment_no, 3, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

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

    result_extractor(experiment_no, 4, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

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

    result_extractor(experiment_no, 5, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

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

    result_extractor(experiment_no, 6, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

    print("-----------------------------------------------------------------------------------------------------------")
    # pipeline 7
    start_time = time.time()
    y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = (
        simple_pipeline(forest.AMFClassifier(seed=seed), None, None, data, target))

    finish_time = time.time()
    total_time = finish_time - start_time
    print("Time of pipeline 7: ", total_time)

    # add the results from the pipeline in the lists
    y_real.append(y_real_temp)
    y_predicted.append(y_predicted_temp)
    data_drifts.append(data_drifts_temp)
    concept_drifts.append(concept_drifts_temp)

    result_extractor(experiment_no, 7, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

    print("-----------------------------------------------------------------------------------------------------------")
    # pipeline 8
    start_time = time.time()
    y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = (
        simple_pipeline(forest.AMFClassifier(seed=seed), preprocessing.StandardScaler, None, data, target))

    finish_time = time.time()
    total_time = finish_time - start_time
    print("Time of the pipeline 8: ", total_time)

    # add the results from the pipeline in the lists
    y_real.append(y_real_temp)
    y_predicted.append(y_predicted_temp)
    data_drifts.append(data_drifts_temp)
    concept_drifts.append(concept_drifts_temp)

    result_extractor(experiment_no, 8, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

    print("-----------------------------------------------------------------------------------------------------------")
    # pipeline 9
    start_time = time.time()
    y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = (
        use_AML4S(data, target, True, True, seed=seed))

    finish_time = time.time()
    total_time = finish_time - start_time
    print("Time of pipeline 9: ", total_time)

    # add the results from the pipeline in the lists
    y_real.append(y_real_temp)
    y_predicted.append(y_predicted_temp)
    data_drifts.append(data_drifts_temp)
    concept_drifts.append(concept_drifts_temp)

    result_extractor(experiment_no, 9, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

    # Delete the temporal variables
    del y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp

    # evaluation of pipelines
    evaluates = evaluation(y_real, y_predicted, metrics.Accuracy())

    # plots for metrics and drifts of the pipelines
    create_plots(evaluates, data_drifts, concept_drifts)

    # plots for data
    data_plot(data, 50)