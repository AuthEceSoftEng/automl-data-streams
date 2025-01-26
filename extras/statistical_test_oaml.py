from scipy.stats import wilcoxon
import pandas as pd
from Functions.Evaluation import evaluation
from river import metrics


def take_saved_data(file_name):
    df = pd.read_csv(file_name)
    df = df.replace({'True': 1, 'False': 0})
    df = df.apply(pd.to_numeric, errors='coerce')
    y_real=df.iloc[:, 0].tolist()
    y_pred=df.iloc[:, 1].tolist()

    return y_real, y_pred
oaml = []
for dataset in ('elec', 'vehicle', 'airlines', 'hyperplane'):
    df = pd.read_csv(f"../More_datasets/OAML-basic {dataset}.csv", delimiter=';')
    oaml.append(df["Y"].values[-1])

y_real = []
y_predicted = []
pip = []
for experiment_no in (4,5,6,7):
    folder = 'Experiments'
    # take the results of the pipeline 1
    y_real_temp, y_predicted_temp = take_saved_data(f"../Experiments/Results/experiment{experiment_no}_pipeline2_results.csv")
    # add the results in lists
    y_real.append([None if pd.isna(x) else x for x in y_real_temp])
    y_predicted.append([None if pd.isna(x) else x for x in y_predicted_temp])

pipeline_2 = []
# evaluation of pipelines
_, evaluates = evaluation(y_real, y_predicted, metrics.Accuracy())
for ev in evaluates:
    pipeline_2.append(ev[-1])

# to run faster comment the above and uncomment the bellow

# oaml=[0.8172, 0.8276, 0.6288, 0.8371]
# pipeline_2=[0.9092773393963711, 0.8543506188951651, 0.6535636155143589, 0.889740766690021]

s = wilcoxon(oaml, pipeline_2, alternative="less", method="approx")
print()
print("Wilcoxon test result")
print(s)
print("Statistic: " + str(s.statistic))
print("p-value: " + str(s.pvalue))
print("z Statistic: " + str(s.zstatistic))
