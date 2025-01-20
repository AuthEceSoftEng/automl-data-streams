from Functions.Evaluation import evaluation
from river import metrics
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel
import numpy as np

def take_saved_data(file_name):

    df = pd.read_csv(f'../Experiments/Results/{file_name}')
    df = df.replace({'True': 1, 'False': 0})
    df = df.apply(pd.to_numeric, errors='coerce')
    y_real=df.iloc[:, 0].tolist()
    y_pred=df.iloc[:, 1].tolist()
    data_drifts= [value for value in df.iloc[:, 2].tolist() if pd.notna(value)]
    concept_drifts=[value for value in df.iloc[:, 3].tolist() if pd.notna(value)]

    return y_real, y_pred, data_drifts, concept_drifts

def wilcoxon_test(pipeline_no1,pipeline_no2,experiment_no):

    # create the lists to save the results
    y_real = []
    y_predicted = []

    # take the results of the pipeline 1
    y_real_temp, y_predicted_temp, _, _ = take_saved_data(f"experiment{experiment_no}_pipeline{pipeline_no1}_results.csv")
    # add the results in lists
    y_real.append(y_real_temp)
    y_predicted.append(y_predicted_temp)

    # take the results of the pipeline 2
    y_real_temp, y_predicted_temp, _, _ = take_saved_data(f"experiment{experiment_no}_pipeline{pipeline_no2}_results.csv")
    # add the results in lists
    y_real.append(y_real_temp)
    y_predicted.append(y_predicted_temp)

    # evaluation of pipelines
    _, evaluates = evaluation(y_real, y_predicted, metrics.Accuracy())
    # print(y_predicted[0][450 :])
    stat, p_value = wilcoxon(evaluates[0][450 :],evaluates[1][450 :])
    print (f"Results of wilcoxon between pipeline {pipeline_no1} and {pipeline_no2} in experiment {experiment_no}: \n", " - statistic: ", stat, "\n  - p_value: ", p_value)

wilcoxon_test(9, 2, 1)
wilcoxon_test(9, 2, 2)
wilcoxon_test(9, 2, 3)