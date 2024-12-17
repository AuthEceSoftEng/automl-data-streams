import pandas as pd
from river import metrics

from Functions.Comparison_with_OAML_basic_plot import compare_with_oaml
from Functions.Create_Plots import create_plots
from Functions.Evaluation import evaluation

def take_saved_data(file_name):

    df = pd.read_csv(f'../Experiments/Results/{file_name}')
    df = df.replace({'True': 1, 'False': 0})
    df = df.apply(pd.to_numeric, errors='coerce')
    y_real=df.iloc[:, 0].tolist()
    y_pred=df.iloc[:, 1].tolist()
    data_drifts= [value for value in df.iloc[:, 2].tolist() if pd.notna(value)]
    concept_drifts=[value for value in df.iloc[:, 3].tolist() if pd.notna(value)]

    return y_real, y_pred, data_drifts, concept_drifts

# create the lists to save the results
y_real = []
y_predicted = []
data_drifts = []
concept_drifts = []

experiment_no = 4

y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = take_saved_data(f"experiment{experiment_no}_pipeline1_results.csv")
# add the results from the pipeline in the lists
y_real.append([None if pd.isna(x) else x for x in y_real_temp])
y_predicted.append([None if pd.isna(x) else x for x in y_predicted_temp])
data_drifts.append(data_drifts_temp)
concept_drifts.append(concept_drifts_temp)

y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = take_saved_data(f"experiment{experiment_no}_pipeline2_results.csv")
# add the results from the pipeline in the lists
y_real.append([None if pd.isna(x) else x for x in y_real_temp])
y_predicted.append([None if pd.isna(x) else x for x in y_predicted_temp])
data_drifts.append(data_drifts_temp)
concept_drifts.append(concept_drifts_temp)

if experiment_no == 4:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('../More_datasets/OAML-basic elec.csv', delimiter=';')
elif experiment_no == 5:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('../More_datasets/OAML-basic vehicle.csv', delimiter=';')
elif experiment_no == 6:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('../More_datasets/OAML-basic airlines.csv', delimiter=';')
elif experiment_no == 7:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('../More_datasets/OAML-basic hyperplane.csv', delimiter=';')
# print(y_real_temp)
# print(y_real[-1])

# Delete the temporal variables
del y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp

# evaluation of pipelines
evaluates = evaluation(y_real, y_predicted, metrics.Accuracy())

# plots for metrics and drifts of the pipelines
create_plots(evaluates, data_drifts, concept_drifts)

result_to_compare = []
# add the mean evaluation of pipelines in a list
for mean_evaluation in evaluates[1]:
    result_to_compare.append(mean_evaluation)
# add the mean evaluation of oaml
result_to_compare.append(df.iloc[:, 0].tolist())
# compare the pipelines with oaml
compare_with_oaml(result_to_compare)
