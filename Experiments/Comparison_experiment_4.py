import pandas as pd
from river import metrics, preprocessing, tree, forest, datasets
from Functions.Comparison_with_OAML_basic_plot import compare_with_oaml
from Functions.Create_Plots import create_plots
from Functions.Data_plot import data_plot
from Functions.Prepare_data import prepare_data
from Functions.Evaluation_sliding_window import evaluation
from AutoML_pipeline.AutoML_Usage import use_automl
import time

from Functions.Result_extractor import result_extractor

"""

    An example of how to use all the functions of the project

"""
if __name__ == "__main__":
    seed = 30
    # Load dataset
    dataset = datasets.Elec2()
    data = prepare_data(dataset)

    # set the target name
    target = 'y'

    # create the lists to save the results
    y_real = []
    y_predicted = []
    data_drifts = []
    concept_drifts = []

    print("-----------------------------------------------------------------------------------------------------------")
    # pipeline 1
    start_time = time.time()
    y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = (
        use_automl(data, target, False, True, seed = seed))

    finish_time = time.time()
    total_time = finish_time - start_time
    print("Time of pipeline 1: ", total_time)

    # add the results from the pipeline in the lists
    y_real.append(y_real_temp)
    y_predicted.append(y_predicted_temp)
    data_drifts.append(data_drifts_temp)
    concept_drifts.append(concept_drifts_temp)

    result_extractor(4, 1, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

    print("-----------------------------------------------------------------------------------------------------------")
    # pipeline 2
    start_time = time.time()
    y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = (
        use_automl(data, target, True, True, seed = seed))

    finish_time = time.time()
    total_time = finish_time - start_time
    print("Time of pipeline 2: ", total_time)

    # add the results from the pipeline in the lists
    y_real.append(y_real_temp)
    y_predicted.append(y_predicted_temp)
    data_drifts.append(data_drifts_temp)
    concept_drifts.append(concept_drifts_temp)

    result_extractor(4, 2, y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp)

    # Delete the temporal variables
    del y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp

    # evaluation of pipelines
    evaluates = evaluation(y_real, y_predicted, metrics.Accuracy())

    # plots for metrics and drifts of the pipelines
    create_plots(evaluates, data_drifts, concept_drifts)

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('../More_datasets/OAML-basic elec.csv', delimiter=';')

    result_to_compare = []
    # add the mean evaluation of pipelines in a list
    for mean_evaluation in evaluates[1]:
        result_to_compare.append(mean_evaluation)
    # add the mean evaluation of oaml
    result_to_compare.append(df.iloc[:, 0].tolist())
    # compare the pipelines with oaml
    compare_with_oaml(result_to_compare)

    # plots for data
    data_plot(data, 50)
