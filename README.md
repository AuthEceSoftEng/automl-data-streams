# automl-data-streams

AutoML Pipeline for Data Streams (AML4S)

## Overview

This repository contains a loan data stream generator and a fully automated online machine learning method for data streams. It also contains visualization tools, experiments and examples of the method.

## Table of contents

- [Manual](#manual)
  - [Usage](#usage) 
  - [Example](#example)
- [Function Details](#function-details)
  - [AML4S](#aml4s_class) 
    - [\_\_init__](#__init__)
    - [init_train](#init_train)
    - [predict_one](#predict_one)
    - [learn_one](#learn_one)
  - [use_AML4S](#use_aml4s)
  - [find_best_pipeline](#find_best_pipeline)
  - [change_pipeline](#change_pipeline)
  - [simple_pipeline](#simple_pipeline)
  - [Conver_arf_to_csv file](#convert_arff_to_csv-file)
  - [create_loandataset](#create_loandataset)
  - [prepare_data](#prepare_data)
  - [evaluation](#evaluation)
  - [create_plots](#create_plots)
  - [Comparison_with_OAML](#comparison_with_oaml)
  - [data_plot](#data_plot)
  - [accuracy_check](#accuracy_check)
  - [split_data](#split_data)
- [Dataset](#dataset)

## Manual

### Usage

- To install AML4S from GitHub use: `git clone https://github.com/AuthEceSoftEng/automl-data-streams.git`
- To create a loan dataset, use the [`create_loandataset`](#ucreate_loandatasetu) function.
- To convert a dataset from arff to csv use the [`convert_arff_to_csv`](#uconvert_arff_to_csv-fileu) function.
- To prepare a dataset for the pipeline (if it’s not a list of dictionaries) from a CSV or from real datasets of River, use the [`prepare_data`](#uprepare_datau) function.
- To create and use an instance of AML4S, use the [`AML4S`] class.
  1. Create an instance of AML4S with [`__init__`](#__init__).
  2. Create a small training data set.
  3. Train AML4S for the first time with [`init_train`](#init_train).
  4. Predict using AML4S with [`predict_one`](#predict_one).
  5. Train AML4S with a new instance with [`learn_one`](#learn_one).
- To evaluate the created pipelines (one or more), use the [`evaluation`](#uevaluationu) function.
- To create plots for the evaluations, use the [`create_plots`](#ucreate_plotsu) function.
- To create plots of dataset features, use the [`data_plot`](#udata_plotu) function.
- To create interactive diagrams from saved files of the experiments with online methods run file [plots_online_exp.py](Plots_from_saved_csv/plots_online_exp.py).
- To create interactive diagrams from saved files of the experiments with OAML run file [plots_oaml_exp.py](Plots_from_saved_csv/plots_oaml_exp.py).
- To create comparison diagrams from saved files of the experiments with online methods run file [comparison_with_online.py](Plots_from_saved_csv/comparison_with_online_algorithms.py).
- To create comparison diagrams from saved files of the experiments with OAML run file [comparison_with_online.py](Plots_from_saved_csv/comparison_with_oaml.py).

### Example

A good example of how to use the AML4S is included in the `AML4S_Usage` [file](AML4S/AML4S_Usage.py).

Some good examples of how to use all the functions are included in the `Exeperiments` [directory](Experiments).

## Function Details

### AML4S_class

- **File:** [`AML4S_class.py`](AML4S/AML4S_class.py)
- **Description:** Contains the functions and the parameters of the AML4S object. 

#### \_\_init__

- **Description:** Creates the object AML4S (constructor).

##### Usage:

```
 AML4S(target, data_drift_detector, consept_drift_detector)
```

##### Arguments:
 
- `target` (str): The target variable for the model to predict.
- `data_drift_detector` (boolean): True if there is data drift detector, else False.
- `consept_drift_detector` (boolean): True if there is concept drift detector, else False.
- `seed` (int | None): Random seed for reproducibility

#### init_train

- **Description:** Trains the pipeline for the first time with a provided dataset.

##### Usage:

```
init_train(self, init_train_data)
```

##### Arguments:

- `init_train_data` (list[dict]): List of dictionaries with the training data.

#### predict_one

- **Description:** Predicts the target variable given the features.  

##### Usage:

```
predict_one(self, x)
```

##### Arguments:

- `x` (dict): Sample of data with the features.

#### Output:

- `y` (int): Predicted target values.

### learn_one

- **Description:** Training sample by sample of the pipeline

##### Usage:

```
learn_one(self, x, y)
```

#### Arguments:

- `x` (dict): Sample of data with the features.
- `y` (int): Predicted target values.
---
### use_AML4S

- **File:** [`AML4S_Usage.py`](AML4S/AML4S_Usage.py)
- **Description:** Executes the AutoML pipeline on the provided dataset, including data drift and concept drift detection.

#### Usage:

```
use_AML4S(data, target, data_drift_detector, consept_drift_detector)
```

#### Arguments:

- `data` (list): The dataset to be processed by the pipeline.
- `target` (str): The target variable for the model to predict.
- `data_drift_detector` (boolean): True if there is data drift detector, else False.
- `consept_drift_detector` (boolean): True if there is concept drift detector, else False.

#### Output:

- `y_real` (list): Real target values.
- `y_pred` (list): Predicted target values.
- `pipeline.data_drifts` (list): Detected data drifts.
- `pipeline.concept_drifts` (list): Detected concept drifts.
---
### find_best_pipeline

- **File:** [`Find_best_pipeline.py`](AML4S/Find_best_pipeline.py)
- **Description:** Finds the best-performing pipeline among various models and configurations, using data and concept drift detection methods.

#### Usage:

```
find_best_pipeline(x_train, y_train, data_drift_detector_method, concept_drift_detector_method)
```

#### Arguments:

- `x_train` (list): Data with feature values for training.
- `y_train` (list): Data with target values for training.
- `data_drift_detector_method` (object): Method for detecting data drift.
- `concept_drift_detector_method` (object): Method for detecting concept drift.

#### Output:

- `pipeline` (object): The selected best pipeline.
- `accuracy` (object): The accuracy of the selected pipeline.
- `data_drift_detectors` (object): Data drift detectors used in the selected pipeline.
- `concept_drift_detector` (object): The concept drift detector used in the selected pipeline.
---
### change_pipeline

- **File:** [`Change_pipeline.py`](AML4S/Change_pipeline.py)
- **Description:** Trains and evaluates a new AutoML pipeline, selecting it if it performs better than the current one.

#### Usage:

```
change_pipeline(pipeline_old, x_train, y_train, data_drift_detectors_old, concept_drift_detector_old, data_drift_detector_method, concept_drift_detector_method, buffer_accuracy)
```

#### Arguments:

- `pipeline_old` (object): The existing classifier pipeline.
- `x_train` (list): Data with feature values for training.
- `y_train` (list): Data with target values for training.
- `data_drift_detectors_old` (object): The existing pipeline's data drift detectors.
- `concept_drift_detector_old` (object): The existing pipeline's concept drift detector.
- `data_drift_detector_method` (object): The method for detecting data drift.
- `concept_drift_detector_method` (object): The method for detecting concept drift.
- `buffer_accuracy` (object): The accuracy of the current model in the buffer.

#### Output:

- `pipeline` (object): The selected pipeline (either new or old).
- `accuracy` (object): The accuracy of the selected pipeline.
- `data_drift_detectors` (object): The data drift detectors used in the selected pipeline.
- `concept_drift_detector` (object): The concept drift detector used in the selected pipeline.
---
### simple_pipeline

- **File:** [`Simple_pipeline_use.py`](Simple_pipeline/Simple_pipeline_use.py)
- **Description:** Constructs a simple machine learning pipeline using a model, an optional preprocessor, and an optional feature selector. It then trains and evaluates the pipeline on the provided dataset.

#### Usage:

```
simple_pipeline(model, preprocessor, feature_selector, data, target)
```

#### Arguments:

- `model` (object): The machine learning model to be used in the pipeline.
- `preprocessor` (object or None): An optional preprocessing object. If `None`, no preprocessing is applied.
- `feature_selector` (object or None): An optional feature selector object. If `None`, no feature selection is applied.
- `data` (list): The dataset to be used for training and prediction. Each element should be a dictionary of features.
- `target` (str): The name of the target variable in the dataset.

#### Output:

- `y_real` (list): The actual target values from the dataset.
- `y_pred` (list): The predicted target values from the pipeline.
- `data_drifts` (list): A placeholder list, empty in this implementation.
- `concept_drifts` (list): A placeholder list, empty in this implementation.

---
### Convert_arff_to_csv file
- **File:** [`Convert_arff_to_csv`](More_datasets/Convert_arff_to_csv.py)
- **Description:** File converter from arff to csv.
 
#### Usage:

```
convert_arff_to_csv('arff_name.arff', 'csv_name.csv')
```

#### Arguments:

- `arff_file` (string): path for arff file e.g. 'arff_name.arff'
- `csv_name` (string): path for new csv file

#### Output:
- saved vsc file
---
### create_loandataset

- **File:** [`Create_loandataset.py`](LoanDataset/Create_loandataset.py)
- **Description:** Creates a loan dataset with specified drifts.

#### Usage:

```
create_loandataset(class_num, datalimit, conceptdriftpoints, datadriftpoints, seed)
```

#### Arguments:

- `class_num` (2, 3, 4): Number of class in the output of the generator
- `datalist` (int): Number of data samples in the dataset (e.g., 30000).
- `conceptdriftpoints` (list[dict]): Points of drifts with function names (e.g., [4000: "crisis", 10000: "normal"]).
- `datadriftpoints` (list[dict]): Points of drifts with function names (e.g., [2000: "crisis", 8000: "normal"]).
- `seed` (int): Seed for dataset reproducibility (e.g., 42).

#### Output:

- `data` (list[dict]): List of dictionaries containing the created dataset.
---
### prepare_data

- **File:** [`Prepare_data.py`](Functions/Data_plot.py)
- **Description:** Prepares the dataset for the pipeline.

#### Usage:

```
prepare_data(dataset)
```

#### Arguments:

- `dataset` (str or River dataset): Path of a CSV file or a River dataset.

#### Output:

- `data` (list[dict]): List of dictionaries with the dataset's data.
---
### evaluation

- **File:** [`Evaluation.py`](Functions/Evaluation.py)
- **Description:** Evaluates the pipelines created.

#### Usage:

```
evaluation(y_real, y_predicted, metric_algorithm)
```

#### Arguments:

- `y_real` (list[list]): Real target values from each pipeline.
- `y_predicted` (list[list]): Predicted target values from each pipeline.
- `metric_algorithm` (object): Instance of the metric for evaluation.

#### Output:

- `results` (list[list]): Evaluation results for each pipeline.
---
### create_plots

- **File:** [`Create_Plots.py`](Functions/Create_Plots.py)
- **Description:** Creates plots for the evaluation metrics of each pipeline.

#### Usage:

```
create_plots(evaluates, data_drifts, concept_drifts)
```

#### Arguments:

- `evaluates` (list[list]): Evaluation results from the `evaluation` function.
- `data_drifts` (list[list]): Data drift points from each pipeline.
- `concept_drifts` (list[list]): Concept drift points from each pipeline.

#### Output:

- Plot of the metric we used in evaluation for all pipelines used.
---
### comparison_with_oaml

- **File:** [`Comparison_with_OAML_basic_plot.py`](Functions/Comparison_with_OAML_basic_plot.py)
- **Description:** Creates plots in same figure to compare metric results of some methods with OAML-basic.  

#### Usage:

```
compare_with_oaml(results)
```

#### Arguments:

- `results` (list[list]): Evaluation results from the `evaluation` function and OAML results with step 1000 and start 6000.

#### Output:

- Figure with the metric plot of every method.
---
### data_plot

- **File:** [`Data_plot.py`](Functions/Data_plot.py)
- **Description:** Creates plots for dataset features.

#### Usage:

```
data_plot(data, step)
```

#### Arguments:

- `data` (list[dict]): List of dictionaries containing the data.
- `step` (int): Step of the visualization of the dataset.

#### Output:

- Plots of each feature in the dataset.
---
### accuracy_check

- **File:** [`Accuracy_check.py`](Functions/Accuracy_check.py)
- **Description:** Compares accuracy against a mean accuracy to decide if a model retrain is needed.

#### Usage:

```
accuracy_check(mean_accuracy, y_true_buffer, y_predicted_buffer)
```

#### Arguments:

- `mean_accuracy` (float): The mean accuracy to compare against.
- `y_true_buffer` (list): True target values of the last samples.
- `y_predicted_buffer` (list): Predicted target values of the last samples.

#### Output:

- `need_change` (boolean): Indicates if the accuracy difference exceeds a threshold.
---
### split_data

- **File:** [`Split_data.py`](Functions/Split_data.py)
- **Description:** Splits the data into features and target.

#### Usage:

```
split_data(dictionary, target_key)
```

#### Arguments:

- `dictionary` (dict): Dictionary containing features and target value.
- `target_key` (string): Name of the target variable.

#### Output:

1. `features` (dict): Features of the input sample.
2. `target` : Target value of the sample.



## Dataset

The generator can produce datasets with data and concept drifts at specified points.
- **File for 2 class output:** [`loandataset_2_class.py`](LoanDataset/loandataset_2_class.py)
- **File for 3 class output:** [`loandataset_3_class.py`](LoanDataset/loandataset_3_class.py)
- **File for 4 class output:** [`loandataset_4_class.py`](LoanDataset/loandataset_4_class.py)
- **Description:** Loandataset generator
### Concept Drift:

- **crisis:** Tighter limits.
- **normal:** Normal limits.
- **growth:** Looser limits.

### Data Drift:

- **crisis:** Smaller salaries.
- **normal:** Normal salaries.
- **growth:** Bigger salaries.

To create a loan dataset, use the [`create_loandataset`](#ucreate_loandatasetu) function.