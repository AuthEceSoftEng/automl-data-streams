import os
import pickle
import numpy as np
import pandas as pd
from river import metrics
import matplotlib.pyplot as plt
from Functions.Evaluation import evaluation

imagespath = r"SET_THIS"
figsize=(9, 3.0)

""" GATHER ALL THE DATA """
for experiment_no in range(4, 8):
    if not os.path.exists('datao' + str(experiment_no) + '.pkl'):

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
        
        # print(y_real_temp)
        # print(y_real[-1])
        
        # Delete the temporal variables
        del y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp
        
        # evaluation of pipelines
        evaluates = evaluation(y_real, y_predicted, metrics.Accuracy())

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
        evaluates[0].append([None] * len(df.iloc[:, 0]))
        evaluates[1].append(df.iloc[:, 0].tolist())
        
        with open('datao' + str(experiment_no) + '.pkl', 'wb') as f:
            pickle.dump([evaluates, data_drifts, concept_drifts], f)


""" CREATE TABLES """
pipelines = ["AML4S with concept drift detection", "AML4S with concept and data drift detection", "OAML-basic"]
pipelines = ["AML4S-CD", "AML4S", "OAML-basic"]

results = []
for i, pipeline in enumerate(pipelines):
    res = [pipeline]
    for experiment_no in range(4, 8):
        with open('datao' + str(experiment_no) + '.pkl', 'rb') as f:
            evaluates, data_drifts, concept_drifts = pickle.load(f)
            res.append("%.2f\\%%" % (100 * evaluates[1][i][-1]))
    results.append(res)

results = pd.DataFrame(results, columns=["Algorithm", "Electricity", "Vehicle", "Airlines", "Hyperplane"])
print(results.to_latex(index = False))


""" CREATE PLOTS """
for experiment_no, experiment_name in zip(range(4, 8), ("electricity", "vehicle", "airlines", "hyperplane")):
    
    with open('datao' + str(experiment_no) + '.pkl', 'rb') as f:
        evaluates, data_drifts, concept_drifts = pickle.load(f)


    """ PLOT AML4S """
    results = evaluates[0] # evaluates[0] is the accuracy, evaluates[1] is the average accuracy
    data_drifts, concept_drifts = data_drifts[-1], concept_drifts[-1] # drifts are only found for AML4S (which is the last method)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(results[1], label = "AML4S", color = "#1f77b4", linestyle = "solid")
    
    ax.vlines(concept_drifts, ymin = 0, ymax = 1, label = "Concept drifts", color = "gray", linestyle = "dashed")
    ax.vlines(data_drifts, ymin = 0, ymax = 1, label = "Data drifts", color = "gray", linestyle = "dotted", linewidth=1.75)
    
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Data instances")
    
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(imagespath, "runoaml" + experiment_name + ".pdf"))
        

    """ PLOT AML4S VS OAML """
    results = evaluates[1] # evaluates[0] is the accuracy, evaluates[1] is the average accuracy
    
    linestyles =['dashed', 'solid', 'dashdot', (5, (10, 3)), (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (5, 5)), 'solid'] #
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#000000'] #, '#bcbd22', '#17becf']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for p, (pipeline, color, linestyle) in enumerate(zip(pipelines, colors, linestyles)):
        if p == 2:
            ax.plot(np.arange(6000, (len(results[p])+5) * 1000 + 1, 1000), results[p], label = pipeline, color = color, linestyle = linestyle)
        else:
            ax.plot(results[p], label = pipeline, color = color, linestyle = linestyle)
    
    #ax.set_ylim(-0.09, 0.85)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Data instances")
    
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(imagespath, "comparisonoaml" + experiment_name + ".pdf"))

#plt.show()
