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

def wilcoxon_test(pipeline_to_compare):

    # create the lists to save the results
    y_real = []
    y_predicted = []

    for experiment_no in (1, 2, 3, 'covtype', 'rialto'):
        folder = 'Experiments'
        if experiment_no not in (1, 2, 3):
            folder = 'extras'
        # take the results of the pipeline 1
        y_real_temp, y_predicted_temp, _, _ = take_saved_data(f"../{folder}/Results/experiment{experiment_no}_pipeline9_results.csv")
        # add the results in lists
        y_real.append([None if pd.isna(x) else x for x in y_real_temp])
        y_predicted.append([None if pd.isna(x) else x for x in y_predicted_temp])

        # take the results of the pipeline 2
        y_real_temp, y_predicted_temp, _, _ = take_saved_data(f"../{folder}/Results/experiment{experiment_no}_pipeline{pipeline_to_compare}_results.csv")
        # add the results in lists
        y_real.append([None if pd.isna(x) else x for x in y_real_temp])
        y_predicted.append([None if pd.isna(x) else x for x in y_predicted_temp])

    # evaluation of pipelines
    _, evaluates = evaluation(y_real, y_predicted, metrics.Accuracy())
    AML4S = []
    other_method = []
    for i in range(0,len(evaluates),2):
        AML4S.append(evaluates[i][-1])
        other_method.append(evaluates[i+1][-1])

    print(AML4S, "\n")
    print(other_method)

    # to run faster uncomment the method you want and the AML4S and comment all the above

    # other_method = [0.9067, 0.83175, 0.7914, 0.7079320220580642, 0.32961702127659576] # for comparison with HoeffdingAdaptiveTreeClassifier with standard scaler
    # other_method = [0.6272, 0.55735, 0.5692, 0.9443402201675697, 0.8180790273556231] # for comparison with AMFClassifier
    # other_method = [0.8565, 0.7718, 0.77195, 0.9312320571692151, 0.8177386018237082] # for comparison with AMFClassifier with standard scaler
    # AML4S = [0.9304859335038363, 0.8450639386189258, 0.8252685421994885, 0.9578270710105036, 0.8203667481662592]

    s = wilcoxon(other_method, AML4S, alternative="less", method="approx")
    print()
    print("Wilcoxon test result")
    print(s)
    print("Statistic: " + str(s.statistic))
    print("p-value: " + str(s.pvalue))
    print("z Statistic: " + str(s.zstatistic))

wilcoxon_test(1)
wilcoxon_test(2)
wilcoxon_test(3)
wilcoxon_test(4)
wilcoxon_test(5)
wilcoxon_test(6)
wilcoxon_test(7)
wilcoxon_test(8)
