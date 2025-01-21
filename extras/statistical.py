from Functions.Evaluation import evaluation
from river import metrics
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel
import numpy as np

def take_saved_data(file_name):

    df = pd.read_csv(file_name)
    df = df.replace({'True': 1, 'False': 0})
    df = df.apply(pd.to_numeric, errors='coerce')
    y_real=df.iloc[:, 0].tolist()
    y_pred=df.iloc[:, 1].tolist()
    data_drifts= [value for value in df.iloc[:, 2].tolist() if pd.notna(value)]
    concept_drifts=[value for value in df.iloc[:, 3].tolist() if pd.notna(value)]

    return y_real, y_pred, data_drifts, concept_drifts

def wilcoxon_test(pipeline_no1,pipeline_no2):

    # create the lists to save the results
    y_real = []
    y_predicted = []

    for experiment_no in (1, 2, 3, 'adult', 'covtype'):
        folder = 'Experiments'
        if experiment_no in ('adult', 'covtype'):
            folder = 'extras'
        # take the results of the pipeline 1
        y_real_temp, y_predicted_temp, _, _ = take_saved_data(f"../{folder}/Results/experiment{experiment_no}_pipeline{pipeline_no1}_results.csv")
        # add the results in lists
        y_real.append([None if pd.isna(x) else x for x in y_real_temp])
        y_predicted.append([None if pd.isna(x) else x for x in y_predicted_temp])

        # take the results of the pipeline 2
        y_real_temp, y_predicted_temp, _, _ = take_saved_data(f"../{folder}/Results/experiment{experiment_no}_pipeline{pipeline_no2}_results.csv")
        # add the results in lists
        y_real.append([None if pd.isna(x) else x for x in y_real_temp])
        y_predicted.append([None if pd.isna(x) else x for x in y_predicted_temp])

    # evaluation of pipelines
    _, evaluates = evaluation(y_real, y_predicted, metrics.Accuracy())
    # print(y_predicted[0][450 :])

    eval1 = [evaluates[0][-1], evaluates[2][-1], evaluates[4][-1], evaluates[6][-1], evaluates[8][-1]]
    eval2 = [evaluates[1][-1], evaluates[3][-1], evaluates[5][-1], evaluates[7][-1], evaluates[9][-1]]
    print (eval1, eval2)

    stat, p_value = wilcoxon(eval1, eval2)
    print (f"Results of two side wilcoxon between pipeline {pipeline_no1} and {pipeline_no2}: \n", " - statistic: ", stat, "\n  - p_value: ", p_value)
    stat, p_value = wilcoxon(eval1, eval2, alternative="greater")
    print(f"Results of one side wilcoxon between pipeline {pipeline_no1} and {pipeline_no2}: \n", " - statistic: ", stat, "\n  - p_value: ", p_value)


wilcoxon_test(9, 2)
