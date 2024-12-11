import os
import pickle
import pandas as pd
from river import metrics
import matplotlib.pyplot as plt
from Functions.Evaluation_sliding_window import evaluation

imagespath = r"SET_THIS"


""" GATHER ALL THE DATA """
for experiment_no in range(1, 4):
    if not os.path.exists('data' + str(experiment_no) + '.pkl'):
        
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
        y_real.append(y_real_temp)
        y_predicted.append(y_predicted_temp)
        data_drifts.append(data_drifts_temp)
        concept_drifts.append(concept_drifts_temp)
        
        y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = take_saved_data(f"experiment{experiment_no}_pipeline2_results.csv")
        # add the results from the pipeline in the lists
        y_real.append(y_real_temp)
        y_predicted.append(y_predicted_temp)
        data_drifts.append(data_drifts_temp)
        concept_drifts.append(concept_drifts_temp)
        
        y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = take_saved_data(f"experiment{experiment_no}_pipeline3_results.csv")
        # add the results from the pipeline in the lists
        y_real.append(y_real_temp)
        y_predicted.append(y_predicted_temp)
        data_drifts.append(data_drifts_temp)
        concept_drifts.append(concept_drifts_temp)
        
        y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = take_saved_data(f"experiment{experiment_no}_pipeline4_results.csv")
        # add the results from the pipeline in the lists
        y_real.append(y_real_temp)
        y_predicted.append(y_predicted_temp)
        data_drifts.append(data_drifts_temp)
        concept_drifts.append(concept_drifts_temp)
        
        y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = take_saved_data(f"experiment{experiment_no}_pipeline5_results.csv")
        # add the results from the pipeline in the lists
        y_real.append(y_real_temp)
        y_predicted.append(y_predicted_temp)
        data_drifts.append(data_drifts_temp)
        concept_drifts.append(concept_drifts_temp)
        
        y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = take_saved_data(f"experiment{experiment_no}_pipeline6_results.csv")
        # add the results from the pipeline in the lists
        y_real.append(y_real_temp)
        y_predicted.append(y_predicted_temp)
        data_drifts.append(data_drifts_temp)
        concept_drifts.append(concept_drifts_temp)
        
        y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = take_saved_data(f"experiment{experiment_no}_pipeline7_results.csv")
        # add the results from the pipeline in the lists
        y_real.append(y_real_temp)
        y_predicted.append(y_predicted_temp)
        data_drifts.append(data_drifts_temp)
        concept_drifts.append(concept_drifts_temp)
        
        y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = take_saved_data(f"experiment{experiment_no}_pipeline8_results.csv")
        # add the results from the pipeline in the lists
        y_real.append(y_real_temp)
        y_predicted.append(y_predicted_temp)
        data_drifts.append(data_drifts_temp)
        concept_drifts.append(concept_drifts_temp)
        
        y_real_temp, y_predicted_temp, data_drifts_temp, concept_drifts_temp = take_saved_data(f"experiment{experiment_no}_pipeline9_results.csv")
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
        
        with open('data' + str(experiment_no) + '.pkl', 'wb') as f:
            pickle.dump([evaluates, data_drifts, concept_drifts], f)


""" CREATE TABLES """
pipelines = ["HoeffdingAdaptiveTreeClassifier", "HoeffdingAdaptiveTreeClassifier with StandardScaler", \
             "HoeffdingTreeClassifier", "HoeffdingTreeClassifier with StandardScaler", "LogisticRegression", \
             "LogisticRegression with StandardScaler", "AMFClassifier", "AMFClassifier with StandardScaler", "AML4S"]
pipelines = ["Hoeffding Adaptive Tree", "Hoeffding Adaptive Tree (S)", \
             "Hoeffding Tree", "Hoeffding Tree (S)", "Logistic Regression", \
             "Logistic Regression (S)", "AMF Classifier", "AMF Classifier (S)", "AML4S"]

results = []
for i, pipeline in enumerate(pipelines):
    res = [pipeline]
    for experiment_no in range(1, 4):
        with open('data' + str(experiment_no) + '.pkl', 'rb') as f:
            evaluates, data_drifts, concept_drifts = pickle.load(f)
            if i == len(pipelines) - 1:
                res.append("\\textbf{%.2f\\%%}" % (100 * evaluates[1][i][-1]))
            else:
                res.append("%.2f\\%%" % (100 * evaluates[1][i][-1]))
    results.append(res)

results = pd.DataFrame(results, columns=["Algorithm", "2 classes", "3 classes", "4 classes"])
print(results.to_latex(index = False))


""" CREATE PLOTS """
for experiment_no in range(1, 4):
    experiment_name = str(experiment_no + 1) + "classes"
    
    with open('data' + str(experiment_no) + '.pkl', 'rb') as f:
        evaluates, data_drifts, concept_drifts = pickle.load(f)
    
    
    """ PLOT AML4S """
    results = evaluates[0] # evaluates[0] is the accuracy, evaluates[1] is the average accuracy
    data_drifts, concept_drifts = data_drifts[-1], concept_drifts[-1] # drifts are only found for AML4S (which is the last method)
    
    fig, ax = plt.subplots(figsize=(9, 3.2))
    
    ax.plot(results[0], label = "AML4S", color = "#1f77b4", linestyle = "solid")
    
    ax.vlines(concept_drifts, ymin = 0, ymax = 1, label = "Concept drifts", color = "gray", linestyle = "dashed")
    ax.vlines(data_drifts, ymin = 0, ymax = 1, label = "Data drifts", color = "gray", linestyle = "dotted", linewidth=1.75)
    
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Data instances")
    
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(imagespath, "runonline" + experiment_name + ".pdf"))
    
    
    """ PLOT AML4S VS ONLINE MODELS """
    results = evaluates[1] # evaluates[0] is the accuracy, evaluates[1] is the average accuracy
    
    linestyles =['dotted', 'dashed', 'dashdot', (5, (10, 3)), (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (5, 5)), 'solid'] #
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#000000'] #, '#bcbd22', '#17becf']
    
    fig, ax = plt.subplots(figsize=(9, 3.2))
    
    for p, (pipeline, color, linestyle) in enumerate(zip(pipelines, colors, linestyles)):
        ax.plot(results[p], label = pipeline, color = color, linestyle = linestyle)
    
    #ax.set_ylim(-0.09, 0.85)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Data instances")
    
    plt.tight_layout()
    plt.legend(loc='lower right', ncol=2)
    plt.savefig(os.path.join(imagespath, "comparisononline" + experiment_name + ".pdf"))
#plt.show()
